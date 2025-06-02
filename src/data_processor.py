import os
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np

from transformers import (RobertaTokenizer)


class DataProcessor(object):
    def __init__(self, data_dir, dataset_name, tokenizer, seed=42):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer 
        self.read_types(os.path.join(self.data_dir, self.dataset_name))

        # random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def read_txt(self, file_dir, dataset_name, data_name):
        file_name = os.path.join(file_dir, dataset_name, f"{data_name}.txt")
        res = []
        sentence = []
        with open(file_name) as fp:
            for line in fp:
                splits = line.strip().split(" ")
                if len(splits) < 2:
                    res.append(sentence)
                    sentence = []
                    continue
                sentence.append(splits)
        return res

    def read_types(self, file_path):
        type_file = open(os.path.join(file_path, 'types.txt'))
        types = [line.strip() for line in type_file.readlines()]
        self.entity_types = []
        for entity_type in types:
            if entity_type != "O":
                self.entity_types.append(entity_type.split('-')[-1])

    def get_label_map(self, tag_scheme='io'):
        label_map = {'O': 0}
        num_labels = 1
        for entity_type in self.entity_types:
            label_map['B-'+entity_type] = num_labels
            if tag_scheme == 'iob':
                label_map['I-'+entity_type] = num_labels + 1
                num_labels += 2
            elif tag_scheme == 'io':
                label_map['I-'+entity_type] = num_labels
                num_labels += 1
        label_map['UNK'] = -100
        inv_label_map = {k: v for v, k in label_map.items()}
        self.label_map = label_map
        self.inv_label_map = inv_label_map
        return label_map, inv_label_map


    def get_train_truth_tensor(self, max_seq_length=128):
        res = []
        with open(os.path.join(self.data_dir, self.dataset_name, "train.txt")) as fp:
            truth = [-100]
            for line in fp:
                splits = line.strip("\n").split(" ")
                if len(splits) < 2:
                    while len(truth) < max_seq_length:
                        truth.append(-100)
                    res.append(truth)
                    truth = [-100]
                    continue
                truth.append(self.label_map[splits[1]])
        return torch.tensor(res, dtype=torch.long)


    def read_file(self, file_dir, dataset_name, data_name):
        file_name = os.path.join(file_dir, dataset_name, f"{data_name}.txt")

        sentences, labels = [], []
        with open(file_name) as fp:
            sentence, label = [], []
            for line in fp:
                splits = line.strip("\n").split(" ")
                if len(splits) < 2:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
                    continue
                sentence.append(splits[0])
                label.append(self.inv_label_map[int(splits[-1])] if splits[-1].isnumeric() else splits[-1])
        return sentences, labels 

    def get_data(self, dataset_name, data_name):
        sentences, labels = self.read_file(self.data_dir, dataset_name, data_name)
        sent_len = [len(sent) for sent in sentences]
        print(f"****** {dataset_name} set stats (before tokenization): sentence length: {np.average(sent_len)} (avg) / {np.max(sent_len)} (max) ******")
        data = []
        for sentence, label in zip(sentences, labels):
            text = ' '.join(sentence)
            label = label
            data.append((text, label))
        return data

    def get_tensor(self, data_name, max_seq_length, drop_o_ratio=0, drop_e_ratio=0):
        data_file = os.path.join(self.data_dir, f"{self.dataset_name}_{data_name}.pt")
        if os.path.exists(data_file) and False:
        # if os.path.exists(data_file):
            print(f"Loading data from {data_file}")
            tensor_data = torch.load(data_file)
        else:
            all_data = self.get_data(dataset_name=self.dataset_name, data_name=data_name)
            raw_labels = [data[1] for data in all_data]
            all_input_ids = []
            all_attention_mask = []
            all_labels = []
            all_valid_pos = []
            for text, labels in tqdm(all_data, desc="Converting to tensors"):
                encoded_dict = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_seq_length, 
                                                          padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')
                input_ids = encoded_dict['input_ids']
                attention_mask = encoded_dict['attention_mask']
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                label_idx = -100 * torch.ones(max_seq_length, dtype=torch.long)
                valid_pos = torch.zeros(max_seq_length, dtype=torch.long)
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                j = 0
                for i, token in enumerate(tokens[1:], start=1):  # skip [CLS]
                    if token == self.tokenizer.sep_token:
                        break
                    if i == 1 or token.startswith('Ġ'):
                        label = labels[j]
                        label_idx[i] = self.label_map[label]
                        valid_pos[i] = 1
                        j += 1
                assert j == len(labels) or i == max_seq_length - 1
                all_labels.append(label_idx.unsqueeze(0))
                all_valid_pos.append(valid_pos.unsqueeze(0))
                
            all_input_ids = torch.cat(all_input_ids, dim=0)
            all_attention_mask = torch.cat(all_attention_mask, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_valid_pos = torch.cat(all_valid_pos, dim=0)
            all_idx = torch.arange(all_input_ids.size(0))
            tensor_data = {"all_idx": all_idx, "all_input_ids": all_input_ids, "all_attention_mask": all_attention_mask, 
                           "all_labels": all_labels, "all_valid_pos": all_valid_pos, "raw_labels": raw_labels}
            print(f"Saving data to {data_file}")
            torch.save(tensor_data, data_file)
        return self.drop_o(tensor_data, drop_o_ratio, drop_e_ratio)

    def drop_o(self, tensor_data, drop_o_ratio=0, drop_e_ratio=0):
        if drop_o_ratio == 0:
            return tensor_data
        labels = tensor_data["all_labels"]
        # neg
        rand_num = torch.rand(labels.size())
        drop_pos = (labels == 0) & (rand_num < drop_o_ratio)
        labels[drop_pos] = -100
        # pos 
        rand_num = torch.rand(labels.size())
        drop_pos = (labels > 0) & (rand_num < drop_e_ratio)
        labels[drop_pos] = -100
        tensor_data["all_labels"] = labels
        return tensor_data


if __name__ == "__main__":
    dir_path = "../data"
    dataset_name = "CoNLL2003_KB"
    pretrained_model = "roberta-base"

    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case=False, cache_dir="./work/LAS/qli-lab/yuepei/bert_model")
    dp = DataProcessor(dir_path, dataset_name, tokenizer)

    # s, l = dp.read_file(f"{dir_path}/CoNLL2003_KB", "test.txt")
    # dp.get_data("CoNLL2003_KB", "test")
    dp.get_label_map()
    tensor_data = dp.get_tensor("train", 128)

    print(tensor_data["all_attention_mask"][0])
    print(tensor_data["all_valid_pos"][0])
    print(tensor_data["all_labels"][0])