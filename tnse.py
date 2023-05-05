# -*- coding:utf-8 -*-
from sklearn import manifold
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from models import ContrastBert
from transformers import AutoTokenizer, AutoModel
import argparse
import os
import pandas as pd
from matplotlib.pyplot import cm


def main_bak(args):
    categories = ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print("*****GPU加载成功*****")
    else:
        device = torch.device("cpu")

    desc_path = os.path.join(args.data_dir, 'Hierarchical_label.csv')
    label_description_df = pd.read_csv(desc_path, encoding='utf-8')
    save_mode = f"bert_{args.loss_type}.bin"
    if args.Contrast:
        save_mode = 'Contrast_' + save_mode
    if args.GCN:
        save_mode = 'GCN_' + save_mode
    if args.LabelEmbedding:
        save_mode = 'LabelEmbedding_' + save_mode

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    model = ContrastBert(config=args)
    model.to(device)
    model_file = os.path.join(args.output_dir, save_mode)
    model.load_state_dict(torch.load(model_file))
    embeddings = []
    for label in categories:
        labels_desc = label_description_df[label_description_df['Section'] == label]['Section_desc'].values[0]
        tokens = labels_desc.split(' ')
        inputs = tokenizer(tokens,
                             padding='max_length',
                             truncation=True,
                             max_length=args.max_seq_length,
                             is_split_into_words=True,
                             add_special_tokens=True,
                             return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _, _, embedding = model(inputs, labels=None, labels_desc_ids=None, dep_type_matrix=None)
            embedding = embedding[0].cpu().detach().numpy()
            embeddings.append(embedding)


    tsne = manifold.TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=50)
    tsne_rep = tsne.fit_transform(embeddings)
    ax = sns.scatterplot(x=tsne_rep[:, 0], y=tsne_rep[:, 1],
                         hue=categories)
    ax.legend(bbox_to_anchor=(1.1, 1.05), loc='upper right')
    plt.savefig(f'./data/{save_mode[:-4]}_category_bak.png')
    plt.close()


def main(args):
    categories = ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print("*****GPU加载成功*****")
    else:
        device = torch.device("cpu")

    desc_path = os.path.join(args.data_dir, 'Hierarchical_label.csv')
    label_description_df = pd.read_csv(desc_path, encoding='utf-8')
    save_mode = f"bert_{args.loss_type}.bin"
    if args.Contrast:
        save_mode = 'Contrast_' + save_mode
    if args.GCN:
        save_mode = 'GCN_' + save_mode
    if args.LabelEmbedding:
        save_mode = 'LabelEmbedding_' + save_mode

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    model = ContrastBert(config=args)
    model.to(device)
    model_file = os.path.join(args.output_dir, save_mode)
    model.load_state_dict(torch.load(model_file))
    embeddings = []
    labels = []
    for label in categories:
        labels_desc_df = label_description_df[label_description_df['Section'] == label]
        labels_desc_df1 = labels_desc_df.drop_duplicates(subset=['Class', 'Class_desc'], keep='first')
        for labels_desc in labels_desc_df1['Class_desc'].values:
            tokens = labels_desc.split(' ')
            inputs = tokenizer(tokens,
                               padding='max_length',
                               truncation=True,
                               max_length=args.max_seq_length,
                               is_split_into_words=True,
                               add_special_tokens=True,
                               return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                _, _, embedding = model(inputs, labels=None, labels_desc_ids=None, dep_type_matrix=None)
                embedding = embedding[0].cpu().detach().numpy()
                embeddings.append(embedding)
                labels.append(label)

    tsne = manifold.TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=50)
    tsne_rep = tsne.fit_transform(embeddings)
    ax = sns.scatterplot(x=tsne_rep[:, 0], y=tsne_rep[:, 1],
                         hue=labels)
    ax.legend(bbox_to_anchor=(1.1, 1.05), loc='upper right')
    plt.savefig(f'./data/{save_mode[:-4]}_category.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--pretrained_model_path", default='bert-base-uncased', type=str,
                        help="Choose bert mode which you need(.bin).")
    parser.add_argument("--output_dir", default='./output/', type=str,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--max_seq_length", default=256, type=int, help="max words length")
    parser.add_argument("--batch_size", default=8, type=int, help="train batch size")
    parser.add_argument("--num_labels", default=62, type=int, help="total labels")
    parser.add_argument('--dropout_prob', default=0.3, type=float, help='drop out probability')
    parser.add_argument("--num_gcn_layers", default=2, type=int, help="layer")
    parser.add_argument("--embedding_size", default=768, type=int, help="embedding size")
    parser.add_argument('--loss_type', default='CE', type=str, help='FL,CE')
    parser.add_argument('--Contrast', default=True, type=bool, help='Contrastive Learning')
    parser.add_argument('--GCN', default=False, type=bool, help='GCN of dep')
    parser.add_argument('--LabelEmbedding', default=False, type=bool, help='LabelEmbedding')
    parser.add_argument('--pre_trained', default=False, type=bool, help='load pre_trained model')
    parser.add_argument('--temp', type=float, default=100, help='temperature for loss function')
    parser.add_argument('--alpha', type=float, default=2, help='alpha for contrast loss function')
    args = parser.parse_args()
    main(args)
