'''
Author: LawsonAbs
Date: 2021-01-21 09:19:23
LastEditTime: 2021-01-22 18:48:40
FilePath: /punLocation/lawson/main.py
'''
import logging
import argparse
from utils import *  # 导入所有的可用函数
# 使用 PreTrainedTokenizer 的原因是：可以使用 return_offset_mapping 函数
from transformers import BertModel,BertTokenizer
from myDataset import MyDataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from pureBert import PureModel
import torch.nn as nn
import torch as t 

logger = logging.getLogger() 
logging.basicConfig(level=logging.INFO)

# 主程序入口
def main():        
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--label_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data's label dir. It's a xml file")
    
    parser.add_argument("--max_seq_len",
                        default=100,
                        type=int,
                        help="the max sequence length")
    
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
    parser.add_argument("--do_train",
                            action='store_true', 
                            help="Whether to run training.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # ==============================准备数据==============================        
    train_data,train_label = readXml(args.data_dir,args.label_dir) # 从文本中获取数据
    import numpy as np
    train_data = np.array(train_data)    
    # train_data ['They hid from the gu...at it out.', "Wal-Mart isn...ing place!', ... ]
    # print(train_data[0])
    label_list = ["O", "P", "X", "[CLS]", "[SEP]"]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 10) # 使用10折交叉分割数据    
    for train_index,test_index  in kf.split(train_data):
        train_examples = list(train_data[train_index])                        
        train_features = convert_examples_to_pron_features(train_examples,
                                                            label_list,
                                                            args.max_seq_len,
                                                            tokenizer)                                                   

        # 为什么这里把数字全部都提取出来了？ =>  使用TensorDataset 方便封装
        all_input_ids = t.tensor([f.input_ids for f in train_features], dtype=t.long)
        all_input_mask = t.tensor([f.input_mask for f in train_features], dtype=t.long)
        all_segment_ids = t.tensor([f.segment_ids for f in train_features], dtype=t.long)
        all_label_ids = t.tensor([f.label_id for f in train_features], dtype=t.long)
        
        # 组合成一个dataset 
        # 看一下这个 TensorDataset 的实现，还是很好玩儿的~
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size) # 去掉了采样器
    

        # ============================== 构造测试数据集 ========
        eval_examples = list(train_data[test_index])        
        eval_features = convert_examples_to_pron_features(eval_examples,
                                                            label_list,
                                                            args.max_seq_len,
                                                            tokenizer)                                                   

        # 为什么这里把数字全部都提取出来了？ =>  使用TensorDataset 方便封装
        all_input_ids = t.tensor([f.input_ids for f in eval_features], dtype=t.long)
        all_input_mask = t.tensor([f.input_mask for f in eval_features], dtype=t.long)
        all_segment_ids = t.tensor([f.segment_ids for f in eval_features], dtype=t.long)
        all_label_ids = t.tensor([f.label_id for f in eval_features], dtype=t.long)
        
        # 组合成一个dataset 
        # 看一下这个 TensorDataset 的实现，还是很好玩儿的~
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=args.train_batch_size) # 去掉了采样器


        # ==============================开始训练==============================
        # 构造模型        
        model = PureModel(args.bert_model,args.max_seq_len) 
        # 设置损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = t.optim.Adam()     
        for index in trange(int(args.num_train_epochs), desc="Train Epoch"):
            tr_loss = 0  # train loss
            nb_tr_examples, nb_tr_steps = 0, 0
            y_true, y_pred = [], []
            logger.info("\n\n----------------------------Start Training ----------------------------")
            for step, batch in enumerate(tqdm(train_dataloader, desc="Train Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, prons_ids, prons_att_mask = batch                
                if not args.do_pron: prons_emb = None
                # 开始执行 model，用于训练
                loss,logits = model(input_ids, segment_ids, input_mask, label_ids)
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
                for i,mask in enumerate(input_mask): #有什么作用？主要是看是不是一句话
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
            logger.info("\n%s", report)# 这里的换行是因为：如果不换行，就会和上面的tqdm输出混一起了。下同
            logger.info("loss: {}".format(tr_loss/nb_tr_examples))
           
            y_pred, y_true = [], []
            logger.info("\n\n----------------------------Start Evaluating----------------------------")
            for input_ids, input_mask, segment_ids, label_ids  in tqdm(eval_dataloader, desc="Evaluating Iterator"):
                
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)
                
            
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
                write_scores(score_file + 'true_'+str(cv_index), y_true) # 最后得到的文件类型是 pickle 
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
                

    # ==============================测试效果==============================

# 将txt文件读出，成一个数组，每项都是一个str
def readTxt(path):
    data = [] # 训练数据 ["I used to be a banker, but I lose interest"...]
    label = [] # 对应的label  ["interest"]
    with open(path,'r') as f:
        line = f.readline()
        line = line.strip()# 去行末换行   
    

if __name__ == "__main__":
    main()