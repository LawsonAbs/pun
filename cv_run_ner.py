from __future__ import absolute_import, division, print_function

import shutil # 删除文件夹
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
    
    '''
    1.--do_train 这一选项更多地是一个标志，而非需要接受一个值的什么东西。
    现在指定一个新的关键词 action， 并赋值为 "store_true"。 
    这意味着，当这一选项存在时，为 args.do_train 赋值为 True。没有指定时则隐含地赋值为 False。
    2.当你为其制定一个值时，它会报错，因为它就是一个标志
    '''
    parser.add_argument("--do_train",
                        action='store_true', # 有了action这个参数，就说明如果
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
        mark = "homo-"  # 有什么用？
    else: 
        mark = "hete-"

    
    score_file = "scores/"+ mark + '/'
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
        logger.info(f"{args.output_dir} already exists. It will be deleted...")
        shutil.rmtree(args.output_dir) # 如果 
        #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    task_name = args.task_name.lower()
    ''' 这个processors 应该是对数据集进行指定格式的处理
    '''
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    
    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    
    all_examples = processor.get_train_examples(args.data_dir)
    all_examples = np.array(all_examples)

    kf = KFold(n_splits=10)
    kf.get_n_splits(all_examples) # ？？？ 这个功能是？

    cv_index = -1  # 什么含义？
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
        model = BertForTokenPronsClassification_v2.from_pretrained(args.bert_model,
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

        # load pretrained embeddings for phonemes
        prons_map = {}
        prons_map, prons_emb = embed_load('./data/pron.'+str(args.pron_emb_size)+'.vec')

        # convert texts to trainable features
        train_features, prons_map = convert_examples_to_pron_features(
            train_examples, label_list, args.max_seq_length, args.max_pron_length, tokenizer, prons_map)
        eval_features, prons_map = convert_examples_to_pron_features(
            eval_examples, label_list, args.max_seq_length, args.max_pron_length, tokenizer, prons_map)
        prons_emb = embed_extend(prons_emb, len(prons_map))
        prons_emb = torch.tensor(prons_emb, dtype=torch.float)

        prons_embedding = torch.nn.Embedding.from_pretrained(prons_emb)
        prons_embedding.weight.requires_grad=False

        # build training set
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps) # 为什么这里把数字全部都提取出来了？ =>  使用TensorDataset 方便封装
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_prons_ids = torch.tensor([f.prons_id for f in train_features], dtype=torch.long)
        all_prons_att_mask = torch.tensor([f.prons_att_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_prons_ids, all_prons_att_mask)

        # 这个采样器需要学习一下
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # build test set
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
        # 总结一下： 不同的train ,eval 步骤都是需要不同的TensoDataset 和 Dataloader 
        model.train()
        best_score = 0
        label_map = {i : label for i, label in enumerate(label_list,1)}

        # start cross-validation training
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
                # 开始执行 model，用于训练
                loss,logits = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask, label_ids)
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
                if (step + 1) % args.gradient_accumulation_steps == 0:  # 这里竟然还设置了一个对梯度累积更新的步骤
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i,mask in enumerate(input_mask): #有什么作用？主要是看是不是一句话，然后将其赋值为 X
                    temp_1 =  []
                    temp_2 = []
                    for j,m in enumerate(mask):
                        if j == 0:
                            continue
                        try:
                            if m and label_map[label_ids[i][j]] != "X":
                                temp_1.append(label_map[label_ids[i][j]])
                                temp_2.append(label_map[logits[i][j]])
                            else:
                                temp_1.pop()
                                temp_2.pop()
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                        except:
                            pass

            report = classification_report(y_true, y_pred, digits=4)
            logger.info("\n%s", report)
            logger.info("loss: {}".format(tr_loss/nb_tr_examples))
           
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
                        logits,att = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask)
                    else:
                        logits = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask)
                
                logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
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
            f1_new = f1_score(y_true, y_pred)
           
            if f1_new  > best_score: 
                best_score = f1_new
                write_scores(score_file + 'true_'+str(cv_index), y_true)
                write_scores(score_file + 'pred_'+str(cv_index), y_pred)
            
        # save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '_' + str(cv_index))
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        label_map = {i : label for i, label in enumerate(label_list,1)}    
        model_config = {"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,"num_labels":len(label_list)+1,"label_map":label_map}
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
        # load a trained model and config that you have fine-tuned
 
    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # 这是读取事先分好的文件作为 dev 数据
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features, prons_map = convert_examples_to_pron_features(
            eval_examples, label_list, args.max_seq_length, args.max_pron_length, tokenizer, prons_map)
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
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
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

            with torch.no_grad():
                if args.do_pron:
                    logits, att = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask)
                else:
                    logits = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask)
            
            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
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
        report = classification_report(y_true, y_pred,digits=4)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)
        

if __name__ == "__main__":
    main()
