from mii import pipeline

model_path = "/root/autodl-tmp/opt-125m"
# model_path = "/root/autodl-tmp/model/checkpoint-final/"  # 不支持

pipe = pipeline(model_path)
prompts = [
    "Hello, what is apple?  ",
    "who are you? ",
    "what's your favourite sports? "
]
outputs = pipe(prompts, max_new_tokens=500)
for output in outputs:
    print("-" * 20 + "\n")
    print("[output: ]", output.generated_text, "\n")