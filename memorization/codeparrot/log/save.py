'''
Save the memorization contents into a csv file.
'''

import os
import argparse
import hashlib
import csv
import re
import pandas as pd
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")
    parser.add_argument('--num_files', type=int, default=53)
    parser.add_argument('--mode', type=str, choices=['analyze', 'extract_prompt', 'all'], default='analyze')
    
    parser.add_argument('--prompt_mode', type=str, default="single_md5",choices=["single_md5","direct_prompt"], help="The mode of the prompt to use for generation")
    parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation(can also be the path to a file containing the prompt)")
    parser.add_argument('--prompt_hash', type=str, default="", help="The hash of the prompt to use for generation")
    
    parser.add_argument('--sample_num', type=int, default=384)
    
    parser.add_argument('--start', type=int, default=0, help="start id")
    parser.add_argument('--end', type=int, default=20000, help="end id")

    args = parser.parse_args()
    return args


def tokenized_text_len(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)


if __name__ == '__main__':
    args=get_args()
    
    folder_path = os.path.join('save', args.model+'-temp'+str(args.temperature)+'-len'+str(args.seq_len)+'-k'+str(args.top_k))
    folder_path = os.path.join(folder_path, f'{args.start}-{args.end -  1}')
    folder_path = os.path.join(folder_path, 'analyze')
    assert os.path.exists(folder_path), f" {folder_path} does not exist"

    pattern = r'>>>>>>>>>>fingerprints (\w+) >>>>>>>>>>>>>(.*?)<<<<<<<<<<fingerprints \1 <<<<<<<<<<'

    # save as csv file
    with open(os.path.join(folder_path, 'all.txt'), 'r') as f_all:
        # process f_all
        content = f_all.read()
        
        memorizations = re.findall(pattern, content, re.DOTALL)
        
        doc_id = 0
        all_memorizations = []
        with open(os.path.join('save', args.model + '_all_memorization.csv'), 'w', newline='\n') as f_all_memorization:
            writer = csv.writer(f_all_memorization)
            writer.writerow(['doc_id', 'MD5', 'corpus', 'text'])
            for memorization in memorizations:
                if '++++fingerprints' in memorization[1]:
                    code = memorization[1].split('++++fingerprints')[0]
                else:
                    code = memorization[1]
                writer.writerow([doc_id, memorization[0], 'codeparrot-clean-train', code])
                all_memorizations.append([doc_id, memorization[0], 'codeparrot-clean-train', code])
                doc_id += 1
        print(f'Obtain {len(all_memorizations)} pieces of memorized data in total.')
        
    tokenizer = AutoTokenizer.from_pretrained('codeparrot/codeparrot')
    
    data = pd.read_csv(os.path.join('save', args.model + '_all_memorization.csv'), lineterminator='\n')
    data.columns = data.columns.str.replace('\r', '')
    data['text_token_num'] = data['text'].apply(tokenized_text_len)

    filtered_data = data[(data['text_token_num'] >= 128) & (data['text_token_num'] <= 512)]
    filtered_data = filtered_data.drop(['text_token_num'], axis=1)
    filtered_data.to_csv(os.path.join('save', args.model + '_filtered_memorization.csv'), index=False, encoding='utf-8')
    print(f'Select {len(filtered_data)} pieces of memorized data of moderate length, not less than 128 tokens.')
