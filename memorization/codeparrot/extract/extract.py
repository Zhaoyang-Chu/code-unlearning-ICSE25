"""
Generate samples and filter out those that are likely to be memorized samples from the training set.
"""

import os
import sys
import json
from tqdm import tqdm
import argparse
import numpy as np
import logging
import hashlib
logging.basicConfig(level='ERROR')
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_model_and_tokenizer(model_name):
    print("Loading model {} ...".format(model_name))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token
    if 'santacoder' in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)  # it seems that santacode cannot be load in half precision
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.config.pad_token_id = model.config.eos_token_id
    
    print("Model {} is loaded.".format(model_name))
    return tokenizer, model


def save_samples(path_to_save: str, text: str, file_id):
    with open(os.path.join(path_to_save, str(file_id)), 'w', encoding='utf-8') as f:
        f.write(text)


def main():
    model_name = args.model
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    tokenizer, model = get_model_and_tokenizer(model_name)
    model.to(device)
    model.eval()

    # set the path to save the generated samples
    path_to_save = 'results/{}-temp{}-len{}-k{}/separate'.format(model_name, args.temperature, args.seq_len, args.top_k)
    os.makedirs(path_to_save, exist_ok=True)
    # set the prompts
    prompts = [tokenizer.bos_token] * args.batch_size
    inputs = tokenizer(prompts, return_tensors="pt")

    print("The generated samples will be saved to {}...".format(path_to_save))
    input_len = len(inputs['input_ids'][0])
    # number of tokens to generate
    seq_len = args.seq_len
    # sample from the top_k tokens output by the model
    top_k = args.top_k
    
    init_existing_count = len(os.listdir(path_to_save))
    existing_count = len(os.listdir(path_to_save))
    num_batches = int(np.ceil(args.N / args.batch_size))
    for i in tqdm(range(num_batches), desc="Generating", position=0):
        if args.temperature > 1.0:
            # use temperature decaying strategy
            start_temperature = 10.0
            end_temperature = 1.0
            decay_tokens = 20
            output_sequences = inputs['input_ids'].to(device)
            with torch.no_grad():
                for step in range(seq_len):
                    outputs = model(output_sequences)
                    logits = outputs.logits[:, -1, :]

                    if step < decay_tokens:
                        decay_ratio = step / decay_tokens
                        current_temperature = start_temperature - (start_temperature - end_temperature) * decay_ratio
                    else:
                        current_temperature = end_temperature

                    logits /= current_temperature
                    probabilities = softmax(logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)

                    output_sequences = torch.cat((output_sequences, next_token), dim=-1)
        else: 
            # batch generation
            output_sequences = model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                top_k=top_k, 
                top_p=1.0
            )

        texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        for text in texts:
            if existing_count >= init_existing_count + args.N:
                break
            save_samples(path_to_save, text, existing_count)  # store the results
            existing_count += 1


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--N', type=int, default=20000, help="Number of samples to generate")
    parser.add_argument('--batch-size', type=int, default=20, help="Batch size for generation")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")
    parser.add_argument('--gpu_id', type=str, default="1", help="specify the GPU id")

    parser.add_argument('--prompt_mode', type=str, default="single_md5", choices=["single_md5", "direct_prompt"], help="The mode of the prompt to use for generation")
    parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation (can also be the path to a file containing the prompt)")
    parser.add_argument('--prompt_hash', type=str, default="", help="The hash of the prompt to use for generation")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
