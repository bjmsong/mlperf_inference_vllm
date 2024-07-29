import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.generation.streamers import BaseStreamer

import pickle
import time
import threading
import tqdm
import queue

import logging
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-70B-SUT")

# greedy search
gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 1,
    "num_beams": 1,
    "do_sample": False
}
vllm_kwargs = {
            "gpu_memory_utilization": 0.95,
            "tensor_parallel_size": 1,
            # "speculative_model": "/root/autodl-tmp/llama2-7b-int4",
            # "num_speculative_tokens": 5,
            # "use_v2_block_manager": True
            "kv_cache_dtype": "fp8_e4m3"
            # "quantization": "gptq" # 这个参数加了反而变慢
}
sampling_params = SamplingParams(min_tokens = 1, max_tokens = 1024, temperature=0, top_p=0.95)

class FirstTokenStreamer(BaseStreamer):
    """ Streams first tokens to a 'holder' """

    def __init__(self, first_token, tokens_cache=[], is_first_token=True, response_ids=[] ):
        """ Response ids added to 'sign' the first token"""

        self.first_token = first_token # Queue for first token
        self.is_first_token = is_first_token

        # Cache for subsequent generated tokens
        self.tokens_cache = tokens_cache

        self.response_ids = response_ids

        self.is_prompt = True # The first tokens sent to the streamer are actually the input prompts

    def put(self, value):
        """ Caches the tokens as they're generated. Assumes bs=1 """

        # Prompts are streamed first so we need to skip the first time value that arrives
        if self.is_prompt:
            self.is_prompt = False
            return

        value = value.item()
        if self.is_first_token:

            # Add generated first token together with its query response_id to first tokens queue
            self.first_token.put((value, self.response_ids[0]))

            self.is_first_token = False
            return

        self.tokens_cache.append(value)


    def end(self):
        pass

    def get_out_tokens(self):
        return self.tokens_cache


class SUT():
    def __init__(self,
                 model_path=None,
                 dtype="bfloat16",
                 device="cpu",
                 batch_size=None,
                 total_sample_count=24576,
                 dataset_path=None,
                 use_cached_outputs=False,  # Set this to True *only for test accuracy runs* in case your prior session was killed partway through
                 workers=1):

        self.model_path = model_path or "meta-llama/Llama-2-70b-chat-hf"
        self.device = device

        if not batch_size:
            if device == "cpu":
                batch_size = 1
            else:
                batch_size = 32  # Reduce to 8 if using 4 GPUs, 16 for 8.
        self.batch_size = batch_size

        # dtype
        # if dtype == 'bfloat16':
        #     self.amp_enabled = True
        #     self.amp_dtype = torch.bfloat16
        # elif dtype == 'float16':
        #     self.amp_enabled = True
        #     self.amp_dtype = torch.float16
        # else:
        #     self.amp_enabled = False
        #     self.amp_dtype = torch.float32

        if 'cuda' in self.device:
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path,
                                   dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count,
                                   device=self.device)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

        self.load_model()

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()  # 队列：线程安全，FIFO

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()
        self.model_lock = threading.Lock()


    def start(self):
        # 多线程意义不大，因为主要瓶颈是GPU
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()


    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """
        
        print(f"{time.ctime()} --- Start to process queries in Thread: {threading.get_ident()} \n")
        
        # 只要子线程不结束，会一直执行下面的代码(轮巡), 一旦query_queue队列里面有数据，就会进行处理
        while True:
            # print("Enter into while true loop: ")
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            fname = "q" + "_".join([str(i) for i in query_ids])
            fname = f"run_outputs/{fname}.pkl"
            _p = Path(fname) # 创建Path对象
            if self.use_cached_outputs and _p.exists():
                # Read cache
                with _p.open(mode="rb") as f:
                    d = pickle.load(f)
                processed_output = d["outputs"]
                tik1 = None
                tik2 = None
                tik3 = None
                tok = None
            else:
                # Construct / collate batch
                max_seq_len = 1024

                tik1 = time.time()

                input_ids = []
                for q in qitem:
                    input_ids.append(self.data_object.input_ids_list[q.index])   
                
                with self.model_lock:
                    pred_output = self.model.generate(prompt_token_ids=input_ids,
                                                   sampling_params=sampling_params)

                tik2 = time.time()

                print(f"avg input length is: {np.mean([len(promot) for promot in input_ids])}") 
                # max_output_len = 1
                # for pred in pred_output:
                #     output_len = len(pred.outputs[0].token_ids)
                #     if max_output_len < output_len:
                #         max_output_len = output_len

                # token_ids_tensor = []
                # input_len = []
                # for pred in pred_output:
                #     token_ids_tensor.append(pad(torch.tensor(pred.outputs[0].token_ids).view(1, -1).to(self.device),
                #                                 (max_output_len - len(pred.outputs[0].token_ids), 0, 0, 0),
                #                                 value=self.tokenizer.pad_token_id))
                #     input_len.append(len(pred.prompt_token_ids))
                # pred_output_tokens = torch.cat(token_ids_tensor)

                # processed_output = self.data_object.postProcess(pred_output_tokens,
                #                                                 input_seq_lens=input_len,
                #                                                 query_id_list=query_ids)

            sum_tokens = 0
            for i in range(len(qitem)):
                n_tokens = len(pred_output[i].outputs[0].token_ids)
                response_array = array.array("B", np.array(pred_output[i].outputs[0].token_ids).tobytes())
                bi = response_array.buffer_info()
                response = [lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)]
                lg.QuerySamplesComplete(response)
                sum_tokens += n_tokens
            # time.sleep(3)
            print(f"avg output length is: {sum_tokens/len(qitem)}") 
            
            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                print(f"Samples run: {self.sample_counter}")
                if tik1:
                    print(f"\tInference time: {tik2 - tik1} s")
                    print(f"\tPostprocess time: {tok - tik2} s")
                    print(f"\t==== Total time: {tok - tik1} s")
                else:
                    print(f"\tLoaded from cache: {_p}")


    def load_model(self):

        self.model = LLM(self.model_path, tokenizer=self.model_path, **vllm_kwargs) 
        
        print("Loaded model")

        self.device = torch.device(self.device)
        if self.device == "cpu":
            self.model = self.model.to(self.device)  # Force CPU if your system has GPU and you specifically want CPU-only run

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loaded tokenizer")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl


    def predict(self,**kwargs):
        raise NotImplementedError


    def issue_queries(self, query_samples):
        """ Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        list_prompts_tokens = []
        list_prompts_attn_masks = []

        print(f"{time.ctime()} --- Main Thread: {threading.get_ident()} \n")
        print(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])  # 每个线程一次处理一个batch_size的query
            query_samples = query_samples[self.batch_size:]
        print(f"IssueQuery done")


    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(self, model_path=None, dtype="bfloat16", device="cpu", total_sample_count=24576, dataset_path=None, batch_size=None, workers=1):

        super().__init__(model_path=model_path, dtype=dtype, device=device, total_sample_count=total_sample_count, dataset_path=dataset_path, workers=workers)

        self.first_token_queue = queue.Queue()

    def start(self):
        
        # Python多线程: 并发而非并行 
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

        # Create first token response thread
        self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
        self.ft_response_thread.start()


    def process_first_tokens(self):

        while True:
            first_token_item = self.first_token_queue.get()

            if first_token_item is None:
                log.info("Exiting First token response thread")
                break

            first_tokens, response_id = first_token_item

            response_data = array.array("B", np.array(first_tokens, np.float32).tobytes())
            bi = response_data.buffer_info()
            response = [lg.QuerySampleResponse(response_id, bi[0], bi[1])]
            lg.FirstTokenComplete(response)

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic """
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = self.data_object.input_ids[qitem.index]
            input_masks_tensor = self.data_object.attention_masks[qitem.index]

            #TODO: This PoC is super slow with significant overhead. Best to create a patch to `generate`
            tokens_cache = []
            tokens_streamer = FirstTokenStreamer(self.first_token_queue, tokens_cache=tokens_cache, is_first_token=True, response_ids=[qitem.id])

            _ = self.model.generate(    input_ids=input_ids_tensor,
                                        attention_mask=input_masks_tensor,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        streamer = tokens_streamer,
                                        **gen_kwargs
                                        )

            output_tokens = tokens_streamer.get_out_tokens()
            n_tokens = len(output_tokens)
            response_array = array.array("B", np.array(output_tokens, np.int32).tobytes())
            bi = response_array.buffer_info()
            response = [lg.QuerySampleResponse(
                qitem.id, bi[0], bi[1], n_tokens)]
            lg.QuerySamplesComplete(response)


    def issue_queries(self, query_samples):

        self.query_queue.put(query_samples[0])


    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()
