"""
A test of Deepseek using torch.

This is a simple prompt-response loop that maintains a conversation history in memory.
"""

import colorama
colorama.just_fix_windows_console()

import torch
from transformers import Qwen2ForCausalLM
from transformers import LlamaTokenizerFast

TEMPERATURE = 0.7
MAX_GENERATIONS_PER_PROMPT = 10
MAX_TOKENS_PER_GENERATION = 1024

device = torch.device("cpu")
torch.set_default_device(device)

tokenizer = LlamaTokenizerFast.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = Qwen2ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").to(dtype=torch.float16)

conversation: list[dict] = []

def user_prefix():
    return colorama.Fore.CYAN + "[user] " + colorama.Style.RESET_ALL

def assistant_prefix():
    return colorama.Fore.MAGENTA + "[assistant] " + colorama.Style.RESET_ALL

with torch.inference_mode():
    while True:
        prompt = input(user_prefix())
        conversation.append({"role": "user", "content": prompt})

        templated = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        outputs = tokenizer(templated, return_tensors="pt")
        
        inference = model.generate(**outputs, temperature=TEMPERATURE, max_new_tokens=MAX_TOKENS_PER_GENERATION, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        response: str = tokenizer.decode(inference[0][len(outputs["input_ids"][0]):])

        conversation.append({"role": "assistant", "content": response})

        if response.endswith(tokenizer.eos_token):
            conversation[-1]["content"] = response[:-len(tokenizer.eos_token)]
            print(f"{assistant_prefix()}{conversation[-1]['content']}")
            continue

        print(f"{assistant_prefix()}{response}", end="")

        num_generations = 0

        while num_generations < MAX_GENERATIONS_PER_PROMPT:
            num_generations += 1
            templated = tokenizer.apply_chat_template(conversation, tokenize=False, continue_final_message=True)

            outputs = tokenizer(templated, return_tensors="pt")

            inference = model.generate(**outputs, temperature=TEMPERATURE, max_new_tokens=MAX_TOKENS_PER_GENERATION, pad_token_id=tokenizer.eos_token_id, do_sample=True)
            added_response = tokenizer.decode(inference[0][len(outputs["input_ids"][0]):])

            if added_response.endswith(tokenizer.eos_token):
                trimmed = added_response[:-len(tokenizer.eos_token)]
                conversation[-1]["content"] += trimmed
                print(trimmed, end="")
                break
            
            conversation[-1]["content"] += added_response
        
        print()
