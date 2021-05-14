from __future__ import absolute_import, division, print_function

import shutil # 删除文件夹
import argparse
import csv
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
# 直接使用 seqeval 来评测

import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defi_num",
                    default=50,
                    type=int,
                    help="The number of definition.")


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
    
    # 我将128 改为了 75。 因为调研后的数据发现最大的长度是68 + cls + sep 也就70
    # 而且超过50 的就只有1 条，所以这里还是只用55的长度，如果超出了，则截断
    parser.add_argument("--max_seq_length",
                        default=55,  
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
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pron",
                        action='store_true',
                        help='Whether to use pronunciation as features.')

    parser.add_argument("--use_sense", # 但是好像在后面没用到
                        action='store_true',
                        help='Whether to use sense as features.')                        
                        
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=5,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
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
    parser.add_argument('--file_suffix',type=int,default=0,
                        help="The suffix of file")

    parser.add_argument('--use_random',
                        action='store_true',
                        default=False, # 默认值为false，即使用 zero 填充
                        help='to judge whether use random number padding sense embedding')


    args = parser.parse_args()
    
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    if "homo" in args.data_dir:
        mark = "homo_"  # 有什么用？用于后面生成文件夹使用
    else: 
        mark = "hete_"
    
    # 这里修改一下文件名
    if args.use_sense:
        mark += "sense_"
    if args.do_pron:
        mark += "pron_"

    score_file = "scores/"+ mark + str(args.file_suffix)+'/'
    if not os.path.isdir(score_file): os.mkdir(score_file)
    args.output_dir = score_file + args.output_dir

    import time
    curTime = time.strftime("%m%d_%H%M%S", time.localtime())
    log_name = curTime + '.log'
    # 根据生成的文件夹，将日志写到其中
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename= "./"+ score_file + log_name, # 以当前时间作为log名，可以指定一个文件夹
                        filemode='w', # 写模式
                        )
    logger = logging.getLogger(__name__)

    logger.info("the paramers in this model are:")
    for k,v in (vars(args).items()):        
        logger.info(f"{k,v}") 

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    
    # 下面这个logger 是从 bert_utils 中导入的
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
    # "X" 是什么意思？  => 如果一个单词被分成了两个token，那么就用x标记
    label_list = processor.get_labels() # ["O", "P", "X", "[CLS]", "[SEP]"]
    
    # bug 源码后面有一个+1，我这里将其去掉了
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    '''
    01.get_train_examples(args.data_dir) 这个是把 发音embedding 也写到了 data example中    
    02. 这里的 arg.data_dir 其实是train.txt 文件。 里面每行的内容如下：
    单词 标记 [发音向量]  => 下面给出示例    
    ...
    to O T,UW1
    a O AH0
    sting P S,T,IH1,NG
    operation O AA2,P,ER0,EY1,SH,AH0,N
    ? O PUNCTUATION_?
    ...

    其实这个将单词映射成发音向量的 步骤是很关键的。否则不知道怎么将向量拼接。
    '''
    all_examples = processor.get_train_examples(args.data_dir)
    all_examples = np.array(all_examples)
    sense_path = "./data/defi_emb_50.txt"
    wordEmb = getAllWordSenseEmb(sense_path) # 得到单词sense 的embedding

    kf = KFold(n_splits=10) # 分割10份
    kf.get_n_splits(all_examples) # ？？？ 这个功能是？ => 感觉像是什么都没有做

    
    '''
    下面就是使用交叉验证将所有的数据分成 train 和 test， 然后进行数据分割
    '''
    cv_index = -1  # 什么含义？ => cross validation index
    from transformers import BertModel
    from transformers import AutoTokenizer # 引入一个包
    auto_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    sense_bert = BertModel.from_pretrained("/home/lawson/pretrain/bert-base-cased")
    sense_bert.to(device) # 放入到gpu中
    for train_index, test_index in kf.split(all_examples):
        cv_index += 1
        train_examples = list(all_examples[train_index])
        eval_examples = list(all_examples[test_index]) # 可以直接在numpy数组中再套一个数组，然后就得到数组的值

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
                  pron_emb_size=args.pron_emb_size, # 16
                  do_pron=args.do_pron,
                  use_sense = args.use_sense, # 是否使用sense
                  defi_num = args.defi_num)
                  
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
            train_examples, 
            label_list,
            args.max_seq_length,
            args.max_pron_length,
            tokenizer,
            prons_map,
            logger)
        eval_features, prons_map = convert_examples_to_pron_features(
            eval_examples,
            label_list,
            args.max_seq_length,
            args.max_pron_length,
            tokenizer,
            prons_map,
            logger)
        prons_emb = embed_extend(prons_emb, len(prons_map))
        prons_emb = torch.tensor(prons_emb, dtype=torch.float)
        # 根据 tensor 创建一个 Embedding 实例
        prons_embedding = torch.nn.Embedding.from_pretrained(prons_emb)
        prons_embedding.weight.requires_grad=False

        # build training set
        logger.info("***** Training Parameters *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)  # 对这个参数不了解
        
        # 为什么这里把数字全部都提取出来了？ =>  使用TensorDataset 方便封装 
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long) # size [1446,max_seq_length]
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_prons_ids = torch.tensor([f.prons_id for f in train_features], dtype=torch.long)
        all_prons_att_mask = torch.tensor([f.prons_att_mask for f in train_features], dtype=torch.long)
        #train_all_sense = getSenseEmbedding(all_input_ids,args.bert_model) # train_sense_emb 
        train_data = TensorDataset(all_input_ids, 
                                    all_input_mask,
                                    all_segment_ids,
                                    all_label_ids,
                                    all_prons_ids, 
                                    all_prons_att_mask)

        # 这个采样器需要学习一下
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # build test set
        logger.info("***** Evaluation Parameters *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_prons_ids = torch.tensor([f.prons_id for f in eval_features], dtype=torch.long)
        all_prons_att_mask = torch.tensor([f.prons_att_mask for f in eval_features], dtype=torch.long)
        
        # 根据 all_input_ids 变换得到 对应的 sense_embedding
        
        eval_data = TensorDataset(all_input_ids, 
                                    all_input_mask,
                                    all_segment_ids,
                                    all_label_ids,
                                    all_prons_ids,
                                    all_prons_att_mask,
                                    )
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # 总结一下： 不同的train ,eval 步骤都是需要不同的TensoDataset 和 Dataloader 
        model.train()
        best_score = 0

        '''原先是从1 开始的字典，因为这个会导致后面 456行的代码出现错误
        temp_2.append(label_map[logits[i][j]]) 
        错误的原因是 logits[i][j] 会出现0，从而导致这种错误
        '''
        label_map = {i : label for i, label in enumerate(label_list,0)}
        id_2_key_map = get_word_key_id_2_map(keyPath = "/home/lawson/program/punLocation/data/key.txt")
        # start cross-validation training
        logger.info("cv: {}".format(cv_index))
        for index in trange(int(args.num_train_epochs), desc="Train Epoch"):
            tr_loss = 0  # train loss
            nb_tr_examples, nb_tr_steps = 0, 0
            y_true, y_pred = [], []
            logger.info("\n\n----------------------------Start Training ----------------------------")
            for step, batch in enumerate(tqdm(train_dataloader, desc="Train Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, prons_ids, prons_att_mask = batch
                # print("\n",input_ids.size()) # torch.size[batch_size,max_seq_length]  
                prons_emb = prons_embedding(prons_ids.detach().cpu()).to(device)
                
                defi_emb = None # 存储一批pun得到的sense embedding
                if args.use_sense : # 如果需要使用sense embedding
                    for input_id in input_ids:
                        tokens = auto_tokenizer.convert_ids_to_tokens(input_id) 
                        cur_pun_emb = getPunEmb(wordEmb,tokens,args.defi_num,args.use_random)
                        # print(cur_pun_emb.size())
                        # size = [word_num * defi_num, defi_dim]
                        cur_pun_emb = cur_pun_emb.view(args.max_seq_length,args.defi_num,768)
                        if defi_emb is None:
                            defi_emb = cur_pun_emb
                        else:
                            defi_emb = torch.cat((defi_emb,cur_pun_emb),0)
                            # defi_emb 的size 是 [batch_size,max_seq_length,defi_num,768]
                
                    defi_emb = defi_emb.cuda()    
                if not args.do_pron: prons_emb = None

                # 开始执行 model 用于训练                
                loss,logits = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask, label_ids, defi_emb)
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
                for i,mask in enumerate(input_mask): 
                    temp_1 = []
                    temp_2 = []
                    for j,m in enumerate(mask): # 只做有文本内容的那部分
                        if j == 0:
                            continue
                        else:
                            temp_1.append(label_map[label_ids[i][j]])
                            temp_2.append(label_map[logits[i][j]])                        
                        
                        # 如果是到pad部分 或者 最后的部分了，那么就break
                        # 最后的部分，说明参数 max_seq_length 设置的不够大
                        if m == 0 or j + 1 == len(mask): 
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
            
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("\n%s", report)# 这里的换行是因为：如果不换行，就会和上面的tqdm输出混一起了。下同
            logger.info("loss: {}".format(tr_loss/nb_tr_examples))
            
            '''
            evaluation 的时候，将源txt文件和 label 一起写入到文件中，这样就方便对照阅读
            '''
            y_pred, y_true = [], [] # 用于保存预测和真实的 label
            all_tokens = [] # 用于保存所有的tokens，然后写入到最后的文件中
            logger.info("\n\n----------------------------Start Evaluating----------------------------")
            for input_ids, input_mask, segment_ids, label_ids, prons_ids, prons_att_mask in tqdm(eval_dataloader, desc="Evaluating Iterator"):
                prons_emb = prons_embedding(prons_ids).to(device)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device) # ？
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                prons_ids = prons_ids.to(device)
                prons_att_mask = prons_att_mask.to(device)
                
                eval_defi_emb = None # 存储一批pun得到的sense embedding          
                if args.use_sense :                
                    for input_id in input_ids:
                        tokens = auto_tokenizer.convert_ids_to_tokens(input_id) 
                        all_tokens.append(tokens)
                        cur_pun_emb = getPunEmb(wordEmb,tokens,args.defi_num,args.use_random)
                        #print(cur_pun_emb.size())
                        # size = [word_num * defi_num, defi_dim]
                        cur_pun_emb = cur_pun_emb.view(args.max_seq_length,args.defi_num,768)
                        if eval_defi_emb is None:
                            eval_defi_emb = cur_pun_emb
                        else:
                            eval_defi_emb = torch.cat((eval_defi_emb,cur_pun_emb),0)
                            # defi_emb 的size 是 [batch_size,max_seq_length,defi_num,768]
                    eval_defi_emb = eval_defi_emb.cuda()
                else: # 依然需要找出原来的tokens
                    for input_id in input_ids:
                        tokens = auto_tokenizer.convert_ids_to_tokens(input_id) 
                        all_tokens.append(tokens)                                        
                if not args.do_pron: prons_emb = None

                with torch.no_grad(): # 不计算梯度
                    logits,att_pron,att_defi = model(input_ids, segment_ids, input_mask, prons_emb, prons_att_mask,defi_emb = eval_defi_emb)
                
                logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()

                for i,mask in enumerate(input_mask):
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
                    # 过滤掉[PAD] 和 [CLS]
                    tokens = [i for i in tokens if i !='[PAD]' and i!='[CLS]']
                    logger.info(f"当前文本信息是：{tokens}")
                    temp_1 =  [] # 作为临时的 true
                    temp_2 = [] # 作为临时的 pred
                    for j,m in enumerate(mask): # 只要 mask 为1 的数据
                        if j == 0: # 跳过CLS 标志
                            continue
                        else:
                            # 并不需要pop操作，直接记录所有的tokens
                            # temp_1.pop()
                            # temp_2.pop()
                            temp_1.append(label_map[label_ids[i][j]])
                            temp_2.append(label_map[logits[i][j]])
                        
                        if m == 0 or j + 1 == len(mask): 
                            temp_1.pop() # pop() 是为了排除SEP 之后的O
                            temp_2.pop()
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            # 从pred 中找出p的标志位，同时去获取 att_defi 的值                            
                            max_value_index = [i for i in range(len (temp_2)) if temp_2[i]=='P']
                            
                            if len(max_value_index) == 1: # 给出完美预测的sense score weight
                                max_value_index = max_value_index[0] + 1 # 默认取第0位，因为之前有CLS向量，所以这里有个+1操作
                                pred_pun_word = tokenizer.convert_ids_to_tokens([input_ids[i][max_value_index].item()])[0] # 得到预测的pun word
                                # 找出该位的attention值 
                                # size = [defi_num]
                                sense_aware = att_defi[i][max_value_index]  
                                # 从sense_aware 中找出与其最相关的top-k个值
                                ind_val_dict = {}
                                for k in range(len(sense_aware)):
                                    ind_val_dict[k] = sense_aware[k]
                                re = list(sorted(ind_val_dict.items(),key=lambda x:x[1],reverse=True))
                                top_k = 5 # 取前5个                                
                                logger.info(f"当前预测得到的双关词是:{pred_pun_word}")
                                
                                for m in range(top_k):
                                    index , weight = re[m]
                                    logger.info(f"index = {index}, weight = {weight},") # 同时打印出key 信息
                                    cur_key = pred_pun_word + "_" + str(index)                                    
                                    if cur_key in id_2_key_map.keys():
                                        key_list = id_2_key_map[cur_key]
                                        logger.info(key_list)
                            break
            
            # 是所有的eval data 完之后，才有这个操作，说明y_true是所有的batch 数据得到的结果
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("\n%s", report)
            f1_new = f1_score(y_true, y_pred)
           
            if f1_new  > best_score:
                best_score = f1_new
                # 这里的 score_file表示的是一个文件名
                write_scores(score_file + 'true_'+str(cv_index), y_true) # 最后得到的文件类型是 pickle 
                write_scores(score_file + 'pred_'+str(cv_index), y_pred)

                '''将源文件和golden label，pred 写在一起'''
                writeToTxt(all_tokens,y_true,y_pred,score_file+"all_"+str(cv_index))                
            
        # save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '_' + str(cv_index))
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
        
        # 不理解为什么这里要从下标1 开始标记 map 
        label_map = {i : label for i, label in enumerate(label_list,1)}
        model_config = {"bert_model":args.bert_model,
                        "do_lower":args.do_lower_case,
                        "max_seq_length":args.max_seq_length,
                        "num_labels":len(label_list)+1,
                        "label_map":label_map
                        }
        json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
        # load a trained model and config that you have fine-tuned
        

if __name__ == "__main__":
    main()