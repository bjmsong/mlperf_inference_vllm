from vllm import LLM, SamplingParams

model_path = "/root/autodl-tmp/model/checkpoint-final/"
model_path = "/root/autodl-tmp/opt-125m"

prompts = [
    "Hello, what is apple?  ",
    "who are you? ",
    "what's your favourite sports? "
]

sampling_params = SamplingParams(
        temperature=0,
        max_tokens=500, 
        frequency_penalty=1.2
        )

llm = LLM(model=model_path, dtype='float16')

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    token_ids = output.outputs[0].token_ids
    print("-" * 20 + "\n")
    print(generated_text, '\n')