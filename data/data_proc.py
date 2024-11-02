#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
from collections import Counter

def main():
    path = './uspto-release'
    data_list = os.listdir(path)
    labels_list = []
    df = pd.DataFrame()
    for files in data_list:
        file_path = os.path.join(path, files)
        data_df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        df = pd.concat([df, data_df])
    for index, row in df.iterrows():
        labels = eval(row['labels'])
        labels_list.extend(labels)
    # 统计各个标签出现的次数
    result = Counter(labels_list)
    print('各个标签出现的次数:', result)
    labels_news=[]
    labels_del = []
    for k, v in result.items():
        if v<500:
            labels_del.append(k)
        else:
            labels_news.append(k)
    dataset = []
    for index, row in df.iterrows():
        la = eval(row['labels'])
        remaining = list(set(la) & set(labels_del))
        if len(remaining)>0:
            continue
        dataset.append(row)
    dataset_df = pd.DataFrame(data=dataset)
    print('样本数：',len(dataset_df))
    labels_list_new = []
    for index, row in dataset_df.iterrows():
        labels = eval(row['labels'])
        labels_list_new.extend(labels)
        # 统计各个标签出现的次数
    result_new = Counter(labels_list_new)
    print('各个标签出现的次数:',len(result_new), result_new)
    labels_list_new = list(set(labels_list_new))
    label_map = {}
    for (i, label) in enumerate(labels_list_new):
        label_map[label] = i
    with open("./label_to_id.json", "w") as f:
        json.dump(label_map, f)
    return dataset_df

def data_split(data):
    from sklearn.utils import shuffle
    data_df = shuffle(data)
    leng = int(len(data_df)*0.8)
    leng1 = int(len(data_df)*0.9)
    train_df = data_df[:leng]
    test_df = data_df[leng:leng1]
    dev_df = data_df[leng1:]
    
    labels_list_new = []
    for index, row in train_df.iterrows():
        labels = eval(row['labels'])
        labels_list_new.extend(labels)
        # 统计各个标签出现的次数
    result_new = Counter(labels_list_new)
    print(len(train_df))
    print('训练集各个标签出现的次数:',len(result_new), result_new)
    train_df.to_csv('./process_data/train.tsv', index=False)
    test_df.to_csv('./process_data/test.tsv', index=False)
    dev_df.to_csv('./process_data/dev.tsv', index=False)


if __name__ == "__main__":
    dataset_df = main()
    data_split(dataset_df)
