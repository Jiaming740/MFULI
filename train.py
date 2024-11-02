import os
import torch
from dataset import load_data
from transformers import AutoTokenizer
from tqdm import trange
import argparse
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from models import ContrastBert, load_model
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
              "ccomp", "clf", "complm", "compound""conj", "cop", "csubj", "csubjpass", "dative", "dep", "det",
              "discourse", "dislocated", "dobj", "expl", "fixed", "flat", "goeswith", "hmod", "hyph", "infmod",
              "intj", "iobj", "list", "mark", "meta", "neg", "nmod", "nn", "npadvmod", "nsubj", "nsubjpass",
              "nounmod", "npmod", "num", "number", "nummod", "oprd", "obj", "obl", "orphan", "parataxis", "partmod",
              "pcomp", "pobj", "poss", "possessive", "preconj", "prep", "prt", "punct", "quantmod", "rcmod", "relcl",
              "reparandum", "root", "vocative", "xcomp"]

def Validation_test(model, test_dataloader, device, hierarchy, top_ks=[1, 5, 10]):
    model.eval()
    # Tracking variables
    nb_test_steps = 0
    all_preds = []
    all_targets = []
    top_k_preds = {k: [] for k in top_ks}
    Loss = 0.0
    for inputs, targets, labels_desc_ids in test_dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)
        labels_desc_ids = {k: v.to(device) for k, v in labels_desc_ids.items()}
        with torch.no_grad():
            logits, loss, _ = model(inputs, labels=targets, labels_desc_ids=labels_desc_ids, hierarchy=hierarchy)
        outputs = torch.sigmoid(logits).cpu().detach().numpy()
        outputs_binary = (outputs > 0.5).astype(int)

        for k in top_ks:
            top_k_output = np.argsort(outputs, axis=1)[:, -k:]
            top_k_preds[k].extend(top_k_output.tolist())

        all_preds.extend(outputs_binary.tolist())
        all_targets.extend(targets.cpu().detach().numpy().tolist())
        Loss += loss.item()
        nb_test_steps += 1

    return Loss / nb_test_steps, all_preds, all_targets, top_k_preds

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
        top_k_accuracy = 0
        for i in range(len(targets)):
            top_k_set = set(top_k_preds[k][i])
            target_set = set(np.nonzero(targets[i])[0])
            top_k_accuracy += len(top_k_set & target_set) > 0
        top_k_accuracies[k] = top_k_accuracy / len(targets)

    return accuracy, micro_f1, macro_f1, macro_precision, macro_recall, micro_precision, micro_recall, ham_loss, top_k_accuracies

def main(args):
    logger.info(f"hyper-parameters:{args}")
    """select GPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("*****GPU加载成功*****")
    else:
        device = torch.device("cpu")
    args.dep_type_num = len(dep_labels)
    label_dict = json.load(open(os.path.join(args.data_dir, 'label_to_id.json'), 'r', encoding='utf-8'))
    save_mode = f"bert_{args.loss_type}.bin"
    if args.Contrast:
        save_mode = 'Contrast_' + save_mode
    if args.LabelEmbedding:
        save_mode = 'LabelEmbedding_' + save_mode

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    model = ContrastBert(config=args, similarity='hierarchical_jaccard').to(device)


    pre_model = os.path.join(args.output_dir, save_mode)
    try:
        load_model(model, pre_model, device, strict=False)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")

    if os.path.exists(pre_model) and args.pre_trained:
        model.load_state_dict(torch.load(pre_model, map_location=device), strict=False)
    bert_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)

    """train and save model"""
    if args.do_train:
        train_dataloader, hierarchy = load_data(mode="train", tokenizer=tokenizer, label_dict=label_dict, args=args)
        dev_dataloader, _ = load_data(mode="dev", tokenizer=tokenizer, label_dict=label_dict, args=args)
        logger.info("***** Running train *****")
        """start training"""
        val_f1 = 0.0
        for num_epoch in trange(args.epochs, desc="Epoch"):
            tr_loss = 0
            tr_steps = 0
            total_steps = len(train_dataloader)
            # Tracking variables
            num_epoch += 1
            for inputs, targets, labels_desc_ids in train_dataloader:
                model.train()
                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = targets.to(device)
                labels_desc_ids = {k: v.to(device) for k, v in labels_desc_ids.items()}
                optimizer.zero_grad()
                logits, loss, _ = model(inputs, labels=targets, labels_desc_ids=labels_desc_ids, hierarchy=hierarchy)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()  # 确保是标量
                tr_steps += 1
                if tr_steps % 500 == 0:
                    logger.info("【train】Epoch:{}, Train_Step={}, Total_Train_Step={}, Train loss: {:.6f}".format(num_epoch,
                                                                                                             tr_steps,
                                                                                                             total_steps,
                                                                                                             tr_loss / tr_steps))
                if tr_steps % 500 == 0:
                    """evaluation model"""
                    dev_loss, dev_outputs, dev_targets, dev_top_k_preds = Validation_test(model, dev_dataloader, device, hierarchy=hierarchy)
                    accuracy, micro_f1, macro_f1, macro_precision, macro_recall, micro_precision, micro_recall, ham_loss, top_k_accuracies = get_metrics(
                        dev_targets, dev_outputs, dev_top_k_preds)
                    logger.info(
                        "【dev】 ham_loss：{:.6f} micro_precision：{:.4f} micro_recall：{:.4f} micro_f1：{:.4f}".format(
                            ham_loss,
                            micro_precision,
                            micro_recall,
                            micro_f1))
                    logger.info(
                        "【dev】accuracy：{:.4f} macro_precision：{:.4f} macro_recall：{:.4f} macro_f1：{:.4f}".format(
                            accuracy,
                            macro_precision,
                            macro_recall,
                            macro_f1))
                    for k, top_k_accuracy in top_k_accuracies.items():
                        logger.info("【dev】 Top-{} accuracy：{:.4f}".format(k, top_k_accuracy))
                    if micro_f1 > val_f1:
                        val_f1 = micro_f1
                        """save model"""
                        output_model_file = os.path.join(args.output_dir, save_mode)
                        torch.save(model.state_dict(), output_model_file)
                        logger.info('******save model*******')

    logger.info("***** training ends *****")
    if args.do_test:
        model_file = os.path.join(args.output_dir, save_mode)
        model.load_state_dict(torch.load(model_file), strict=False)
        logger.info("***** Running evaluation *****")
        """load data"""
        test_dataloader, hierarchy = load_data(mode="test", tokenizer=tokenizer, label_dict=label_dict,
                                               args=args)
        test_loss, test_outputs, test_targets, test_top_k_preds = Validation_test(model, test_dataloader,
                                                                                  device,
                                                                                  hierarchy=hierarchy)

        accuracy, micro_f1, macro_f1, macro_precision, macro_recall, micro_precision, micro_recall, ham_loss, top_k_accuracies = get_metrics(
            test_targets, test_outputs, test_top_k_preds)
        logger.info(
            "【test】 ham_loss：{:.6f} micro_precision：{:.4f} micro_recall：{:.4f} micro_f1：{:.4f}".format(
                ham_loss,
                micro_precision,
                micro_recall,
                micro_f1))
        logger.info(
            "【test】accuracy：{:.4f} macro_precision：{:.4f} macro_recall：{:.4f} macro_f1：{:.4f}".format(
                accuracy,
                macro_precision,
                macro_recall,
                macro_f1))
        for k, top_k_accuracy in top_k_accuracies.items():
            logger.info("【test】 Top-{} accuracy：{:.4f}".format(k, top_k_accuracy))

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
    parser.add_argument("--batch_size", default=8, type=int, help="train batch size")
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
    parser.add_argument('--pre_trained', default=True, type=bool, help='load pre_trained model')
    parser.add_argument('--temp', type=float, default=300, help='temperature for loss function')
    args = parser.parse_args()
    main(args)
