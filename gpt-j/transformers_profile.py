# -*- coding: utf-8 -*-
# Please pull the latest code to run the profiling.
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed
from tqdm import tqdm

seed = 1024
max_experiment_times = 1
context_length_per_experiment = 1
generate_length_per_experiment = 2048
# context_length_per_experiment = 1
# generate_length_per_experiment = 8192
# context_length_per_experiment = 2048
# generate_length_per_experiment = 1
use_flash_attn = True
# fp32 without quantization, bf16 without quantization or int4 quantization (updated on 8.28: we have updated the quantization strategy to AutoGPTQ in this script)
quant_type = "bf16" # fp32, bf16 or int4

set_seed(seed)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

if quant_type == "bf16":
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat", 
        device_map="cuda:0", 
        trust_remote_code=True, 
        bf16=True, 
        use_flash_attn=use_flash_attn
    ).eval()
elif quant_type == "fp32":
    assert use_flash_attn is False, "FP32 profiling cannot be performed with Flash-Attention activated."
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat", 
        device_map="cuda:0", 
        trust_remote_code=True, 
        fp32=True, 
        use_flash_attn=False
    ).eval()    
elif quant_type == "int4":
    # please install AutoGPTQ following the readme to use quantization
    from auto_gptq import AutoGPTQForCausalLM
    model = AutoGPTQForCausalLM.from_quantized(
        "Qwen/Qwen-7B-Chat-Int4", 
        device="cuda:0", 
        trust_remote_code=True, 
        use_safetensors=True, 
        use_flash_attn=use_flash_attn
    ).eval()

# Specify hyperparameters for generation
config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
config.min_length = generate_length_per_experiment + context_length_per_experiment
config.max_new_tokens = generate_length_per_experiment

time_costs = []
context_str = '我' * context_length_per_experiment
max_gpu_memory_cost = 0
for _ in tqdm(range(max_experiment_times)):
    inputs = tokenizer(context_str, return_tensors='pt')
    inputs = inputs.to(model.device)
    t1 = time.time()
    pred = model.generate(**inputs, generation_config=config)
    time_costs.append(time.time() - t1)
    assert pred.shape[1] == config.min_length
    max_gpu_memory_cost = max(max_gpu_memory_cost, torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()

print("Average generate speed (tokens/s): {}".format((max_experiment_times * generate_length_per_experiment) / sum(time_costs)))
print(f"GPU Memory cost: {max_gpu_memory_cost / 1024 / 1024 / 1024}GB")
print("Experiment setting: ")
print(f"seed = {seed}")
print(f"max_experiment_times = {max_experiment_times}")
print(f"context_length_per_experiment = {context_length_per_experiment}")
print(f"generate_length_per_experiment = {generate_length_per_experiment}")
print(f"use_flash_attn = {use_flash_attn}")
print(f"quant_type = {quant_type}")