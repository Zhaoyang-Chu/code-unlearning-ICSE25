import argparse
import feather
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smoothie = SmoothingFunction().method4


# calculate BLEU-4 score
def calc_bleu4(tokenizer, sample, generated):
    ref = tokenizer.decode(sample)
    hyp = tokenizer.decode(generated)
    return sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)


def memorization_extraction(args):
    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        pad_token_id=tokenizer.eos_token_id
    )
    model.resize_token_embeddings(len(tokenizer))
    if args.fp16:
        model.half()
    model.to(device)

    df = feather.read_dataframe('benchmark.feather')
    print(df)
    df['prefix'] = df['sample'].apply(lambda x: x[:100])

    gen_suffix = []
    # iterate with batch size
    with torch.no_grad():
        for i in tqdm(range(0, len(df), args.batch_size)):
            batch = torch.tensor(df.iloc[i: i + args.batch_size].prefix.tolist()).to(device)
            output = model.generate(batch, max_length=150)[..., 100:].tolist()
            gen_suffix.extend(output)

    df['gen_suffix'] = gen_suffix
    df['bleu4'] = df.apply(lambda x: calc_bleu4(tokenizer, x['suffix'], x['gen_suffix']), axis=1)
    
    memorization_df = df[df['bleu4'] == 1]
    print(memorization_df)
    print(memorization_df.columns)
    memorization_df.rename(columns={'index': 'doc_id'}, inplace=True)
    memorization_df['text'] = memorization_df['sample'].apply(lambda x: tokenizer.decode(x))
    memorization_df['corpus'] = 'BigQuery'
    memorization_df = memorization_df[['doc_id', 'hash', 'copies', 'corpus', 'text']]
    print(memorization_df)
    model_name = args.model_name_or_path.split('/')[-1]
    memorization_df.to_csv(f'{model_name}_memorization.csv', index=False, encoding='utf-8')


def main():
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Salesforce/codegen-350M-mono", type=str,
                        help="Model to train and evaluate, provide a repo name in Hugging Face hub or a local path.")
    parser.add_argument('--gpu_id', type=str, default="0", help="specify the GPU id")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use fp16 model precision.")
    args = parser.parse_args()
    
    memorization_extraction(args)


if __name__ == '__main__':
    main()
