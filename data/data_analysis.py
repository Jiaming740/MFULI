#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import json
from collections import Counter
import re
import matplotlib
import matplotlib.pyplot as plt

def words_count(row):
    try:
        title = row['title'].replace(';','').replace('.','').replace(':','').replace(',','')
        abstract = row['abstract'].replace(';','').replace('.','').replace(':','').replace(',','')
        abstract_list = re.split(r"[ ]+", abstract)
        title_list = re.split(r"[ ]+", title)
        words_num = len(abstract_list) + len(title_list)
    except Exception as e:
        words_num = 0
    return words_num

def main():
    path = './process_data'
    data_list = os.listdir(path)
    Hierarchical_label = []
    labels_list = []
    df = pd.DataFrame()
    for files in data_list:
        file_path = os.path.join(path, files)
        print(file_path)
        data_df = pd.read_csv(file_path, encoding='utf-8')
        df = pd.concat([df, data_df])
    df['labels_num'] = df['labels'].apply(lambda row: len(eval(row)))
    df['words_num'] = df.apply(lambda row: words_count(row), axis=1)
    print('平均标签个数：', df['labels_num'].mean())
    print('最大标签个数：', df['labels_num'].max())
    print('最小标签个数：', df['labels_num'].min())
    print('平均单词个数：', df['words_num'].mean())
    print('最大单词个数：', df['words_num'].max())
    print('最小单词个数：', df['words_num'].min())
    print(df['words_num'].describe())
    for index, row in df.iterrows():
        labels = eval(row['labels'])
        labels_list.extend(labels)
    # 统计各个标签出现的次数
    result = Counter(labels_list)
    print('各个标签出现的次数:', result)
    labels_list = list(set(labels_list))
    label_map = {}
    for (i, label) in enumerate(labels_list):
        label_map[label] = i
    with open("./label_to_id_bak.json", "w") as f:
        json.dump(label_map, f)
    print('总共涉及标签数量：', len(labels_list))
    print('总共涉及标签类别：', labels_list)
    # 标签数量-专利的直方图
    fig, ax = plt.subplots()
    df.hist('labels_num', ax=ax)
    fig.savefig('labels_num.png', dpi=100, bbox_inches='tight')
    # 专利-单词数量的直方图
    fig, ax = plt.subplots()
    df.hist('words_num', ax=ax)
    fig.savefig('words_num.png', dpi=100, bbox_inches='tight')
    print('-------------------------------')
    # plt.bar(range(len(result.keys())), result.values(), color='skyblue', tick_label=result.keys())
    fig1, ax = plt.subplots()
    plt.bar(range(len(result.keys())), result.values(), color='skyblue')
    fig1.savefig('labels_count.png')

if __name__ == "__main__":
    main()