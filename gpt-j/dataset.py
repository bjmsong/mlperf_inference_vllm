import os
import time
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tokenizer_GPTJ import get_transformer_autotokenizer
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io
import utils
import copy

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class Dataset():
    def __init__(self, dataset_path, batch_size=1, pad_val=1, pad_max=196, total_count_override=None, 
        perf_count_override=None, model_path=None):
        print("Constructing QSL")

        self.dataset = "cnn_dailymail"
        self.model_name = "EleutherAI/gpt-j-6B"
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max

        self.tokenizer = get_transformer_autotokenizer(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.list_data_dict = utils.jload(self.dataset_path)[:total_count_override]

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # 在字符串中查找花括号 {} 中的占位符，并使用提供的字典中对应的键值对来替换占位符
        self.sources = [prompt_input.format_map(
            example) for example in self.list_data_dict]
        self.targets = [
            f"{example['output']}" for example in self.list_data_dict]

        self.source_encoded_input_ids, self.source_encoded_attn_masks = self.encode_samples()

        self.count = total_count_override or len(self.sources)
        self.perf_count = perf_count_override or self.count

    def encode_samples(self):
        print("Encoding Samples")

        total_samples = len(self.sources)

        source_encoded_input_ids = []
        source_encoded_attn_masks = []
        # max_length = 0
        for i in range(total_samples):
            # 都按照max_length的来padding
            # 1. 显存浪费巨大，容易OOM：KV Cache大小跟input length成正比
            # 2. 计算浪费？
            source_encoded = self.tokenizer(self.sources[i], return_tensors="pt",
                                            padding="max_length", truncation=True,
                                            max_length=1919)
            # if(source_encoded.input_ids.shape[1] > max_length):
            #     max_length = source_encoded.input_ids.shape[1]
            source_encoded_input_ids.append(source_encoded.input_ids)
            source_encoded_attn_masks.append(source_encoded.attention_mask)
        # print(max_length)
        return source_encoded_input_ids, source_encoded_attn_masks

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        print("Finished destroying QSL.")