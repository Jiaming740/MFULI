import os
import torch
from dataset import load_data
from transformers import AutoTokenizer
from tqdm import trange
import argparse
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from models import ContrastBert
import logging
import warnings

warnings.filterwarnings("ignore")
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

dep_labels = ["acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc",
              "ccomp", "clf", "complm", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det",
              "discourse", "dislocated", "dobj", "expl", "fixed", "flat", "goeswith", "hmod", "hyph", "infmod",
              "intj", "iobj", "list", "mark", "meta", "neg", "nmod", "nn", "npadvmod", "nsubj", "nsubjpass",
              "nounmod", "npmod", "num", "number", "nummod", "oprd", "obj", "obl", "orphan", "parataxis", "partmod",
              "pcomp", "pobj", "poss", "possessive", "preconj", "prep", "prt", "punct", "quantmod", "rcmod", "relcl",
              "reparandum", "root", "vocative", "xcomp"]


def Validation_test(model, test_dataloader, device, hierarchy, top_ks=[1, 5, 10]):
    model.eval()
    nb_test_steps = 0
    all_preds, all_targets = [], []
    top_k_preds = {k: [] for k in top_ks}
    total_loss = 0.0

    for inputs, targets, labels_desc_ids in test_dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)
        labels_desc_ids = {k: v.to(device) for k, v in labels_desc_ids.items()}

        with torch.no_grad():
            logits, loss, _ = model(inputs, labels=targets, labels_desc_ids=labels_desc_ids, hierarchy=hierarchy)

        outputs = torch.sigmoid(logits).cpu().numpy()
        outputs_binary = (outputs > 0.5).astype(int)

        for k in top_ks:
            top_k_output = np.argsort(outputs, axis=1)[:, -k:]
            top_k_preds[k].extend(top_k_output.tolist())

        all_preds.extend(outputs_binary.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
        total_loss += loss.item()
        nb_test_steps += 1

    avg_loss = total_loss / nb_test_steps
    return avg_loss, all_preds, all_targets, top_k_preds


def get_metrics(targets, outputs, top_k_preds, top_ks=[1, 5, 10]):
    accuracy = accuracy_score(targets, outputs)
    ham_loss = hamming_loss(targets, outputs)
    micro_precision = precision_score(targets, outputs, average='micro')
    micro_recall = recall_score(targets, outputs, average='micro')
    micro_f1 = f1_score(targets, outputs, average='micro')
    macro_f1 = f1_score(targets, outputs, average='macro')
    macro_precision = precision_score(targets, outputs, average='macro')
    macro_recall = recall_score(targets, outputs, average='macro')

    top_k_accuracies = {}
    for k in top_ks:
        top_k_accuracy = sum(
            [len(set(top_k_preds[k][i]) & set(np.nonzero(targets[i])[0])) > 0 for i in range(len(targets))])
        top_k_accuracies[k] = top_k_accuracy / len(targets)

    return accuracy, micro_f1, macro_f1, macro_precision, macro_recall, micro_precision, micro_recall, ham_loss, top_k_accuracies


def main(args):
    logger.info(f"hyper-parameters: {args}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.dep_type_num = len(dep_labels)
    label_dict = json.load(open(os.path.join(args.data_dir, 'label_to_id.json'), 'r', encoding='utf-8'))

    save_mode = f"Contrast_{args.loss_type}.bin" if args.Contrast else f"bert_{args.loss_type}.bin"
    save_mode = f"LabelEmbedding_{save_mode}" if args.LabelEmbedding else save_mode

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    model = ContrastBert(config=args, similarity='hierarchical_jaccard').to(device)

    pre_model = os.path.join(args.output_dir, save_mode)
    if args.pre_trained and os.path.exists(pre_model):
        model.load_state_dict(torch.load(pre_model, map_location=device), strict=False)

    bert_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if args.do_train:
        train_dataloader, hierarchy = load_data(mode="train", tokenizer=tokenizer, label_dict=label_dict, args=args)
        dev_dataloader, _ = load_data(mode="dev", tokenizer=tokenizer, label_dict=label_dict, args=args)

        logger.info("***** Running training *****")
        best_f1 = 0.0
        for epoch in trange(args.epochs, desc="Epoch"):
            model.train()
            tr_loss, tr_steps = 0.0, 0

            for inputs, targets, labels_desc_ids in train_dataloader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = targets.to(device)
                labels_desc_ids = {k: v.to(device) for k, v in labels_desc_ids.items()}

                optimizer.zero_grad()
                _, loss, _ = model(inputs, labels=targets, labels_desc_ids=labels_desc_ids, hierarchy=hierarchy)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()
                tr_steps += 1

                if tr_steps % 500 == 0:
                    logger.info(
                        f"【train】Epoch:{epoch + 1}, Train_Step={tr_steps}, Train loss: {tr_loss / tr_steps:.6f}")

                    dev_loss, dev_outputs, dev_targets, dev_top_k_preds = Validation_test(model, dev_dataloader, device,
                                                                                          hierarchy=hierarchy)
                    accuracy, micro_f1, macro_f1, macro_precision, macro_recall, micro_precision, micro_recall, ham_loss, top_k_accuracies = get_metrics(
                        dev_targets, dev_outputs, dev_top_k_preds)

                    logger.info(
                        f"【dev】 ham_loss：{ham_loss:.6f} micro_precision：{micro_precision:.4f} micro_recall：{micro_recall:.4f} micro_f1：{micro_f1:.4f}")
                    logger.info(
                        f"【dev】accuracy：{accuracy:.4f} macro_precision：{macro_precision:.4f} macro_recall：{macro_recall:.4f} macro_f1：{macro_f1:.4f}")
                    for k, top_k_accuracy in top_k_accuracies.items():
                        logger.info(f"【dev】 Top-{k} accuracy：{top_k_accuracy:.4f}")

                    if micro_f1 > best_f1:
                        best_f1 = micro_f1
                        torch.save(model.state_dict(), pre_model)
                        logger.info('****** Model Saved *******')

    if args.do_test:
        model.load_state_dict(torch.load(pre_model), strict=False)
        logger.info("***** Running evaluation *****")
        test_dataloader, hierarchy = load_data(mode="test", tokenizer=tokenizer, label_dict=label_dict, args=args)
        test_loss, test_outputs, test_targets, test_top_k_preds = Validation_test(model, test_dataloader, device,
                                                                                  hierarchy=hierarchy)

        accuracy, micro_f1, macro_f1, macro_precision, macro_recall, micro_precision, micro_recall, ham_loss, top_k_accuracies = get_metrics(
            test_targets, test_outputs, test_top_k_preds)
        logger.info(
            f"【test】 ham_loss：{ham_loss:.6f} micro_precision：{micro_precision:.4f} micro_recall：{micro_recall:.4f} micro_f1：{micro_f1:.4f}")
        logger.info(
            f"【test】accuracy：{accuracy:.4f} macro_precision：{macro_precision:.4f} macro_recall：{macro_recall:.4f} macro_f1：{macro_f1:.4f}")
        for k, top_k_accuracy in top_k_accuracies.items():
            logger.info(f"【test】 Top-{k} accuracy：{top_k_accuracy:.4f}")

    logger.info("***** evaluation ends *****")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--hierarchy_path", default='./data/Hierarchical_label.csv', type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--pretrained_model_path", default='bert-base-uncased', type=str,
                        help="Choose bert mode which you need(.bin).")
    parser.add_argument("--output_dir", default='./output/', type=str,
                        help="The output directory where the model checkpoints will be written")
    parser.add_argument("--max_seq_length", default=256, type=int, help="max words length")
    parser.add_argument("--batch_size", default=16, type=int, help="train batch size")
    parser.add_argument("--epochs", default=6, type=int, help="run epochs")
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--do_train", default=True, action='store_true', help="train mode")
    parser.add_argument("--do_test", default=True, action='store_true', help="test mode")
    parser.add_argument("--num_labels", default=62, type=int, help="total labels")
    parser.add_argument('--dropout_prob', default=0.2, type=float, help='drop out probability')
    parser.add_argument("--embedding_size", default=768, type=int, help="embedding size")
    parser.add_argument("--weight_decay", default=0.00005, type=int, help="weight_decay")
    parser.add_argument('--loss_type', default='CE', type=str, help='cross_entropy')
    parser.add_argument('--Contrast', default=True, type=bool, help='Contrastive Learning')
    parser.add_argument('--LabelEmbedding', default=True, type=bool, help='LabelEmbedding')
    parser.add_argument('--pre_trained', default=False, type=bool, help='load pre_trained model')
    parser.add_argument('--temp', type=float, default=300, help='temperature for loss function')
    args = parser.parse_args()
    main(args)
