#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import json
import spacy
import re
spacy_nlp = spacy.load("en_core_web_sm")

def clean_str(string):
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r";", " ", string)
    string = re.sub(r":", " ", string)
    string = re.sub(r"-", " ", string)
    string = re.sub(r"'", " ", string)
    return string.strip().lower()

def words_seg(row):
    try:
        Section_desc = row['Section_desc']
        Class_desc = row['Class_desc']
        Subclass_desc = row['Subclass_desc']
        text = Section_desc+' '+Class_desc+' '+Subclass_desc
        text = clean_str(text)
        labels_desc = ' '.join([token.text for token in spacy_nlp(text)])
    except Exception as e:
        labels_desc = ''
        print(e)
    return labels_desc

def main():
    path = './CPCTitleList202208'
    data_list = os.listdir(path)
    Hierarchical_label = []
    label_description = {}
    for files in data_list:
        file_path = os.path.join(path, files)
        print(file_path)
        f = open(file_path, 'r', encoding='utf-8')
        origin_txt = f.readlines()
        f.close()
        for line in origin_txt:

            lines = line.strip().split('\t')
            label = lines[0]
            label_name = lines[-1].lower()
            if len(label)>4:
                continue
            if len(label)==4:
                Hierarchical_label_dict = {}
                Hierarchical_label_dict['Section'] =label[0]
                Hierarchical_label_dict['Class'] =label[:3]
                Hierarchical_label_dict['Subclass'] =label
                Hierarchical_label.append(Hierarchical_label_dict)
            if label not in label_description:
                label_description[label] = label_name
    with open("./label_description.json", "w") as f:
        json.dump(label_description, f)
    Hierarchical_label_df = pd.DataFrame(data=Hierarchical_label)
    Hierarchical_label_df['Section_desc'] = Hierarchical_label_df['Section'].apply(lambda row: label_description[row])
    Hierarchical_label_df['Class_desc'] = Hierarchical_label_df['Class'].apply(lambda row: label_description[row])
    Hierarchical_label_df['Subclass_desc'] = Hierarchical_label_df['Subclass'].apply(lambda row: label_description[row])
    Hierarchical_label_df['labels_desc'] = Hierarchical_label_df.apply(lambda row: words_seg(row), axis=1)
    Hierarchical_label_df['words_num'] = Hierarchical_label_df['labels_desc'].apply(lambda row: len(row.split(' ')))
    Hierarchical_label_df.to_csv('./Hierarchical_label.csv', index=False)


if __name__ == "__main__":
    main()