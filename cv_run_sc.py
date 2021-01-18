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

from bert_models import BertForTokenClassification, BertForSequencePronsClassification_v2, BertForSequencePronsClassification_v3, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report, f1_score, recall_score, precision_score
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from bert_utils import *
from sklearn.model_selection import KFold



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

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

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    if "homo" in args.data_dir:
        mark = "homo-"
    elif "hete" in args.data_dir: 
        mark = "hete-"

    score_file = "scores/" + mark + '/'
    if not os.path.isdir(score_file): os.mkdir(score_file)
    args.output_dir = score_file + args.output_dir

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    
    all_examples = processor.get_train_examples(args.data_dir)
    all_examples = np.array(all_examples)

    kf = KFold(n_splits=10)
    kf.get_n_splits(all_examples)

    cv_index = -1

    for train_index, test_index in kf.split(all_examples):

        cv_index += 1

        train_examples = list(all_examples[train_index])
        eval_examples = list(all_examples[test_index])

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
        model = BertForSequencePronsClassification_v3.from_pretrained(args.bert_model,
                  cache_dir=cache_dir,
                  num_labels=num_labels,
                  max_seq_length=args.max_seq_length,
                  max_prons_length=args.max_pron_length, 
                  pron_emb_size=args.pron_emb_size,
                  do_pron=args.do_pron,
                  device=device)

        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        prons_map = {}
        prons_map, prons_emb = embed_load('./data/pron.'+str(args.pron_emb_size)+'.vec')

        train_features, prons_map = convert_examples_to_pron_SC_features(
            train_examples, label_list, args.max_seq_length, args.max_pron_length, tokenizer, prons_map)
        eval_features, prons_map = convert_examples_to_pron_SC_features(
            eval_examples, label_list, args.max_seq_length, args.max_pron_length, tokenizer, prons_map)
        print(prons_map)
        prons_emb = embed_extend(prons_emb, len(prons_map))
        prons_emb = torch.tensor(prons_emb, dtype=torch.float)
    
        prons_embedding = torch.nn.Embedding.from_pretrained(prons_emb)
        prons_embedding.weight.requires_grad=False

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_prons_ids = torch.tensor([f.prons_id for f in train_features], dtype=torch.long)
        all_prons_att_mask = torch.tensor([f.prons_att_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_prons_ids, all_prons_att_mask)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

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

        model.train()
        best_score = 0
        label_map = {i : label for i, label in enumerate(label_list)}

        logger.info("cv: {}".format(cv_index))
        for index in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            y_true, y_pred = [], []

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, prons_ids, prons_att_mask = batch
                prons_emb = prons_embedding(prons_ids.detach().cpu()).to(device)
                if not args.do_pron: prons_emb = None
                loss,logits,att = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                logits = torch.argmax(F.log_softmax(logits,dim=-1),dim=-1)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                y_pred.extend(logits)
                y_true.extend(label_ids)

            f1_score, recall_score, precision_score  = f1_2d(y_true, y_pred)

            logger.info("precision, recall, f1: {}, {}, {}".format(precision_score, recall_score, f1_score))
            print("loss: {}".format(tr_loss/nb_tr_examples))
           
            y_pred, y_true = [], []
            logger.info("Evaluating...")

            for input_ids, input_mask, segment_ids, label_ids, prons_ids, prons_att_mask in tqdm(eval_dataloader, desc="Evaluating"):
                prons_emb = prons_embedding(prons_ids).to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                prons_ids = prons_ids.to(device)
                prons_att_mask = prons_att_mask.to(device)

                if not args.do_pron: prons_emb = None

                with torch.no_grad():
                    if args.do_pron:
                        logits, att = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask)
                    else:
                        logits = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask)

                logits = torch.argmax(F.log_softmax(logits,dim=-1),dim=-1)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                y_pred.extend(logits)
                y_true.extend(label_ids)

            f1_score, recall_score, precision_score  = f1_2d(y_true, y_pred)
            logger.info("precision, recall, f1: {}, {}, {}".format(precision_score, recall_score, f1_score))
           
            if f1_score  > best_score: 
                best_score = f1_score
                write_scores(score_file + 'true_'+str(cv_index), y_true)
                write_scores(score_file + 'pred_'+str(cv_index), y_pred)
            
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '_' + str(cv_index))
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        label_map = {i : label for i, label in enumerate(label_list)}    
        model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":len(label_list),"label_map":label_map}
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
        # Load a trained model and config that you have fine-tuned

    
if __name__ == "__main__":
    main()
