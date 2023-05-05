# coding=utf-8
import sys
import re
from sklearn.utils import shuffle
import pandas as pd
import os
import json
import torch
from functools import partial
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
import random
import spacy
from dep_parser import *
nlp = spacy.load("en_core_web_sm")


def load_data(mode, tokenizer, label_dict, args):
    collate_fn = partial(my_collate, tokenizer=tokenizer, args=args)
    desc_path = os.path.join(args.data_dir, 'Hierarchical_label.csv')
    label_description_df = pd.read_csv(desc_path, encoding='utf-8')
    if mode == 'train':
        file_path = os.path.join(args.data_dir, 'process_data', 'train.tsv')
        data_df = pd.read_csv(file_path, encoding='utf-8')
        dataset = MyDataset(data_df, label_dict, args, label_description=label_description_df)
        if args.Contrast:
            data_sampler = HierarchicalBatchSampler(batch_size=args.batch_size, dataset=dataset)
            dataloader = DataLoader(dataset, batch_sampler=data_sampler, collate_fn=collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    elif mode == 'test':
        file_path = os.path.join(args.data_dir, 'process_data', 'test.tsv')
        data_df = pd.read_csv(file_path, encoding='utf-8')
        dataset = MyDataset(data_df, label_dict, args, label_description=label_description_df)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    elif mode == 'dev':
        file_path = os.path.join(args.data_dir, 'process_data', 'test.tsv')
        data_df = pd.read_csv(file_path, encoding='utf-8')
        dataset = MyDataset(data_df, label_dict, args, label_description=label_description_df)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,  collate_fn=collate_fn)
    else:
        raise ValueError('unknown mode')

    return dataloader


class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size, dataset, drop_last=False):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.epoch=0
        self.drop_last = drop_last
        self.total_size = len(self.dataset)

    def random_unvisited_sample(self, labels, labels_index_dict, visited, remaining):
        label = random.choice(labels)
        idx_list = labels_index_dict[label]
        remaining_idx_list = list(set(idx_list).difference(visited))
        if len(remaining_idx_list):
            idx = random.choice(remaining_idx_list)
        else:
            idx = remaining[torch.randint(len(remaining), (1,))]
        return idx

    def __iter__(self):
        # g = torch.Generator()
        # g.manual_seed(self.epoch)
        # indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices = torch.randperm(len(self.dataset)).tolist()
        batch = []
        visited = set()
        assert len(indices) == self.total_size
        labels_index_dict = self.dataset.labels_index
        remaining = list(set(indices).difference(visited))
        while len(remaining) > self.batch_size:
            idx = indices[torch.randint(len(indices), (1,))]
            batch.append(idx)
            visited.add(idx)
            tokens, label_id, labels, labels_desc_list, dep_type_matrix = self.dataset[idx]
            labels_index = self.random_unvisited_sample(labels, labels_index_dict, visited, remaining)
            batch.extend([labels_index])
            visited.update([labels_index])
            remaining = list(set(indices).difference(visited))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
            remaining = list(set(indices).difference(visited))

        if len(remaining) > 0 and not self.drop_last:
            batch.extend(list(remaining))
            yield batch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total_size // self.batch_size


def data_processors(raw_data, label_dict, args, label_description=None):
    dataset_list = list()
    labels_index = {}
    for index, row in raw_data.iterrows():
        try:
            title = row['title']
            abstract = row['abstract']
            text = title + '. ' + abstract
            text_new = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", text)
            doc = nlp(text_new)
            tokens = [token.text for token in doc]
            labels = eval(row['labels'])
            label_id = [0] * len(label_dict)
            labels_desc_list = []
            dep_instance_parser = DepInstanceParser(basicDependencies=doc, tokens=tokens)
            dep_type_matrix = dep_instance_parser.get_adj_with_value_matrix(args.max_seq_length)
            for label in labels:
                if label not in labels_index:
                    labels_index[label] = [index]
                else:
                    labels_index[label].append(index)
                id = label_dict[label]
                label_id[id] = 1
                labels_desc = label_description[label_description['Subclass'] == label]['labels_desc'].values[0]
                labels_desc_list.extend(labels_desc.split(' '))

            dataset_list.append((tokens, label_id, labels, labels_desc_list, dep_type_matrix))
        except Exception as e:
            print(e)
            continue
    return dataset_list, labels_index


def my_collate(batch, tokenizer, args):
    tokens, label_ids, labels, labels_desc_list, dep_type_matrix = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding='max_length',
                         truncation=True,
                         max_length=args.max_seq_length,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    labels_desc_ids = tokenizer(labels_desc_list,
                         padding='max_length',
                         truncation=True,
                         max_length=int(args.max_seq_length/2),
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    label_ids = torch.tensor(label_ids, dtype=torch.float)
    dep_type_matrix = torch.tensor(dep_type_matrix, dtype=torch.long)
    return text_ids, label_ids, labels_desc_ids, dep_type_matrix


class MyDataset(Dataset):
    def __init__(self, raw_data, label_dict, args, label_description=None):
        self.dataset = list()
        self.labels_index = {}
        self.args = args
        self.dataset, self.labels_index = data_processors(raw_data, label_dict, args, label_description)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


