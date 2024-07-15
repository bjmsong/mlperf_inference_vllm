 # Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/root/autodl-tmp/model/checkpoint-final/"
model_path = "/root/autodl-tmp/opt-125m"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).cuda().half()

prompts = [
    # "Hello, what is apple?  ",
    # "who are you? "
    "what's your favourite sports? "
]

input_ids = tokenizer(prompts, return_tensors="pt").input_ids.cuda()

generated_ids = model.generate(input_ids, do_sample=False, repetition_penalty=1.2, max_new_tokens=500)

texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(texts)