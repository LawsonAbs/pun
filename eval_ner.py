from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F


from bert_models import BertForTokenPronsClassification_v2, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from bert_utils import *
from sklearn.model_selection import KFold
from seqeval.metrics import classification_report, f1_score, recall_score, precision_score


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The pre-trained model.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_pron_length",
                        default=5,
                        type=int,
                        help="The maximum total input sequence length after pronounciation tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--weight",
                        default=5,
                        type=str)
    parser.add_argument("--pron_emb_size",
                        default=16,
                        type=int,
                        help="The embedding size of pronounciation embedding.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pron",
                        action='store_true',
                        help='Whether to use pronunciation as features.')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    output_model_file = os.path.join(args.model_dir, WEIGHTS_NAME+'_'+str(args.weight))
    output_config_file = os.path.join(args.model_dir, CONFIG_NAME)
    config = BertConfig(output_config_file)
    model = BertForTokenPronsClassification_v2(config, 
              num_labels=num_labels,
              max_seq_length=args.max_seq_length,
              max_prons_length=args.max_pron_length, 
              pron_emb_size=args.pron_emb_size,
              do_pron=args.do_pron,
              device=device)
    model.load_state_dict(torch.load(output_model_file))

    prons_map = {}
    prons_map, prons_emb = embed_load('./data/pron.'+str(args.pron_emb_size)+'.vec')

    eval_examples = processor.get_train_examples(args.data_dir)
    eval_features, prons_map = convert_examples_to_pron_features(
        eval_examples, label_list, args.max_seq_length, args.max_pron_length, tokenizer, prons_map)

    prons_emb = embed_extend(prons_emb, len(prons_map))
    prons_emb = torch.tensor(prons_emb, dtype=torch.float)
    prons_embedding = torch.nn.Embedding.from_pretrained(prons_emb)
    prons_embedding.weight.requires_grad=False

    model.to(device)

    logger.info("***** Running evaluation *****") 
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_prons_ids = torch.tensor([f.prons_id for f in eval_features], dtype=torch.long)
    all_prons_att_mask = torch.tensor([f.prons_att_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_prons_ids, all_prons_att_mask)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    y_true = []
    y_pred = []
    label_map = {i : label for i, label in enumerate(label_list,1)}
    for input_ids, input_mask, segment_ids, label_ids, prons_ids, prons_att_mask in tqdm(eval_dataloader, desc="Evaluating"):
        prons_emb = prons_embedding(prons_ids).to(device)
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        prons_ids = prons_ids.to(device)
        prons_att_mask = prons_att_mask.to(device)

        if not args.do_pron: prons_emb = None
        
        logits,att = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask)
        
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        input_ids = input_ids.to('cpu').numpy()
        prons_att_mask = prons_att_mask.to('cpu').numpy() #(batch_size, seq_len, pron_len)
        att = att.detach().cpu().numpy() #(batch_size, seq_len, pron_len)
        prons_ids = prons_ids.to('cpu').numpy()

        visualize_local(logits, label_ids, input_ids, prons_ids, prons_att_mask, att, label_map, prons_map, tokenizer)

        for i,mask in enumerate(input_mask):
            temp_1 =  []
            temp_2 = []
            for j,m in enumerate(mask):
                if j == 0:
                    continue
                if m and label_map[label_ids[i][j]] != "X":
                    temp_1.append(label_map[label_ids[i][j]])
                    temp_2.append(label_map[logits[i][j]])
                else:
                    temp_1.pop()
                    temp_2.pop()
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break

    report = classification_report(y_true, y_pred, digits=4)
    logger.info("\n%s", report)

if __name__ == "__main__":
    main()
