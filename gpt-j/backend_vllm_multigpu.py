import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlperf_loadgen as lg
from tqdm import tqdm
from accelerate import disk_offload
from vllm import LLM, SamplingParams

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 128,
    "min_new_tokens": 30,
    "num_beams": int(os.environ.get("GPTJ_BEAM_SIZE", "1")), # only beam_size 4 is allowed for official submission
}
sampling_params = SamplingParams(early_stopping=True, use_beam_search=True, temperature=0, 
            best_of = int(os.environ.get("GPTJ_BEAM_SIZE", "4")), min_tokens = 30, max_tokens = 128)

class SUT_base():
    def __init__(self, model_path, dtype, dataset_path, scenario, max_examples, use_gpu=False, network=None, qsl=None):
        self.network = network
        self.model_name = "EleutherAI/gpt-j-6B"
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.dataset_path = dataset_path
        self.max_examples = max_examples
        self.scenario = scenario
        self.qsl = qsl
        self.batch_size = 350
        print("Loading PyTorch model...")
            
        # dtype
        if dtype == 'bfloat16':
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            print("BF16 autocast")
        elif dtype == 'float16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32
        try:
            self.model = LLM(self.model_path, tokenizer=self.model_path, dtype=dtype, tensor_parallel_size=2)
        except ValueError as e: 
            if "disk_offload" in str(e):
                print("Offloading the whole model to disk...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    low_cpu_mem_usage=True if not self.use_gpu else False,
                    torch_dtype=self.amp_dtype,
                    offload_state_dict = True if not self.use_gpu else False
                ).cpu()
                disk_offload(model=self.model, offload_dir="offload")

        # construct SUT
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

    def issue_queries(self, query_samples):
        print("Number of Samples in query_samples : ", len(query_samples))

        for i in tqdm(range(len(query_samples)//self.batch_size)):
            # Activates only when scenario is Offline and network mode is None
            batch_query_samples = query_samples[i*self.batch_size: (i+1)*self.batch_size]
            index_list = [batch_query_samples[i].index for i in range(self.batch_size)]
            query = {
                "query_id": [batch_query_samples[i].id for i in range(self.batch_size)],
                # "inputs_list": [self.qsl.data_object.sources[index] for index in index_list],
                # "target_list": [self.qsl.data_object.targets[index] for index in index_list],
                "input_ids_list": [self.qsl.data_object.source_encoded_input_ids[index] for index in index_list] 
            }
            
            self.inference_call(query)

    def inference_call(self, query):
        ''' Common for all scenarios '''
        # output_batch = self.model.generate(prompts=query["inputs_list"], sampling_params=sampling_params)
        # 会显示batch处理的进度
        output_batch = self.model.generate(prompt_token_ids=query["input_ids_list"], sampling_params=sampling_params)
        
        # with open("output.txt", mode='a') as f:
        #     for i in range(self.batch_size):
        #         f.write("Prompt: \n")
        #         f.write(query["inputs_list"][i] + "\n")
        #         f.write("Target: \n")
        #         f.write(query["target_list"][i] + "\n")
        #         f.write("Output:\n")
        #         f.write(output_batch[i].outputs[0].text + "\n")
        #         f.write("-" * 20 + "\n")  

        # 这个循环跟generate()比，耗时很少，不值得优化
        print("Start submit")
        for i in range(self.batch_size):
            response_text = output_batch[i].outputs[0].text
            pred_output_batch = np.array(output_batch[i].outputs[0].token_ids)

            # Loadgen monitors the response in GPT_QDL
            if self.network == "sut":
                return {"pred_output_batch":pred_output_batch.tolist(), "response_text": response_text}

            response_array = array.array("B", pred_output_batch.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query["query_id"][i], bi[0], bi[1])
            lg.QuerySamplesComplete([response])
        print("End submit")

    def flush_queries(self):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, scenario, max_examples, use_gpu, network, qsl):
        SUT_base.__init__(self, model_path, dtype, dataset_path, scenario, max_examples, use_gpu, network, qsl)
    '''IssueQuery and inference methods implemented in Base class'''


class SUT_Server(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, scenario, max_examples, use_gpu, network, qsl):

        SUT_base.__init__(self, model_path, dtype, dataset_path, scenario, max_examples, use_gpu, network, qsl)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):
        # The issue queries function is called multiple times by the loadgen as per Poisson Distribution
        print("Number of Samples in query_samples : ", len(query_samples))

        index = query_samples[0].index
        input_ids_tensor = self.qsl.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.qsl.data_object.source_encoded_attn_masks[index]
        text = self.qsl.data_object.sources[index]
        query = {
            "input_ids_tensor": input_ids_tensor.tolist(),
            "input_masks_tensor": input_masks_tensor.tolist()
        }
        pred_output_batch = self.inference_call(query, query_samples[0].id).cpu().numpy()
        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)

        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)

class SUT_SingleStream(SUT_base):
    def __init__(self, model_path, dtype, dataset_path, scenario, max_examples, use_gpu, network, qsl):
        SUT_base.__init__(self, model_path, dtype, dataset_path, scenario, max_examples, use_gpu, network, qsl)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        self.total_samples_done = 0

    def issue_queries(self, query_samples):
        # This function is called by the loadgen after completing the previous query
        # 每次一个query
        print("Number of Samples in query_samples : ", len(query_samples))

        index = query_samples[0].index
        input_ids_tensor = self.qsl.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.qsl.data_object.source_encoded_attn_masks[index]
        query = {
            "input_ids_tensor": input_ids_tensor.tolist(),
            "input_masks_tensor": input_masks_tensor.tolist()
        }

        pred_output_batch = self.inference_call(
            query, query_samples[0].id).cpu().numpy()

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)





def get_SUT(model_path, scenario, dtype, dataset_path, max_examples, use_gpu=False, network=None, qsl=None):
    if scenario == "Offline":
        return SUT_Offline(model_path, dtype, dataset_path, scenario, max_examples, use_gpu, network, qsl)
    elif scenario == "Server":
        return SUT_Server(model_path, dtype, dataset_path, scenario, max_examples, use_gpu, network, qsl)
    elif scenario == "SingleStream":
        return SUT_SingleStream(model_path, dtype, dataset_path, scenario, max_examples, use_gpu, network, qsl)