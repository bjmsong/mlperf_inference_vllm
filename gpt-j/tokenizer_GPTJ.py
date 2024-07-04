from transformers import AutoTokenizer

def get_transformer_autotokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,)