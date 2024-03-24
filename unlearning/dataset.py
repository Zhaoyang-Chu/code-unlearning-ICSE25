import pandas as pd
import torch
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(self, tokenizer, dataset_name, type_path, input_length, output_length, args):
        self.args = args
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_name = dataset_name
        self.type_path = type_path

        self.dataset = pd.read_csv(dataset_name, lineterminator='\n')
        self.dataset.columns = self.dataset.columns.str.replace('\r', '')
        if self.type_path == 'train':
            batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.ngpu
            if len(self.dataset) != batch_size:
                raise Exception("Effective batch size should be the same as length of train set.")

    def convert_to_features(self, example_batch):
        doc_id = torch.tensor(example_batch['doc_id'], dtype=torch.int)
        input_, target_ = example_batch['text'], example_batch['text']
        
        source = self.tokenizer(input_, max_length=self.input_length, padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer(target_, max_length=self.output_length, add_special_tokens=False, padding='max_length', truncation=True, return_tensors="pt")
        
        return source, targets, doc_id

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        source, targets, doc_id = self.convert_to_features(data)

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "doc_id": doc_id}

    def __len__(self):
        return len(self.dataset)
