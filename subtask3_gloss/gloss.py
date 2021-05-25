'''
使用pun + gloss 做二分类，然后找出最相思的两个答案
'''
import sys
sys.path.append(r"/home/lawson/program/punLocation/") # 引入当前目录作为模块，否则下面两个模块无法导入
from visdom import Visdom # 可视化输出loss
from subtask3 import attention
from nltk.corpus import wordnet as wn
import random
from subtask3.preprocess import *
import  xml.dom.minidom as dom
import logging
from tqdm import tqdm
import torch.nn as nn
from transformers import BertModel,BertTokenizer
import torch as  t
from torch.utils.data import Dataset,DataLoader
import argparse

tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")
logger = logging.getLogger("gloss")
import time
curTime = time.strftime("%m%d_%H%M%S", time.localtime())
log_name = curTime + '.log'
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    filename="/home/lawson/program/punLocation/log/" + log_name, # 以当前时间作为log名，可以指定一个文件夹
                    filemode='w', # 写模式
                    )
viz = Visdom()
win = "train_loss_"
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--batch_size",
                    default=None,
                    type=int,
                    required=True,
                    help="batch_size")

parser.add_argument("--train_epoch",
                    default=10,
                    type=int,
                    required=True)


parser.add_argument("--eval_batch_size",
                    default=1,
                    type=int,
                    required=True)                        

parser.add_argument("--seed",
                    default=42,
                    type=int,
                    required=True)  

parser.add_argument("--max_length",
                    default=10,
                    type=int,
                    required=True)  

class MyModel(nn.Module):
    def __init__(self,in_fea,out_fea):
        super(MyModel,self).__init__()
        self.linear = nn.Linear(in_fea,out_fea)        
        self.bert = BertModel.from_pretrained("/home/lawson/pretrain/bert-base-cased")
        self.dropout = nn.Dropout(0.1)

    def forward(self,input_ids,attention_mask,token_type_ids):        
        out = self.bert(input_ids,attention_mask,token_type_ids)
        last_hidden_states = out['last_hidden_state'][:,0]
        last_hidden_states = self.dropout(last_hidden_states)
        logits = self.linear(last_hidden_states)
        return logits


class MyDataset(Dataset):
    def __init__(self,train_puns,train_labels):
        super(MyDataset).__init__()
        self.puns = train_puns
        self.labels = train_labels
    
    def __len__(self):
        return len(self.puns)

    def __getitem__(self, index):
        return self.puns[index],self.labels[index]




args = parser.parse_args()


def main():
    pun_words = getAllPunWords(dataPath="/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml")
    id_puns_map = getAllPuns(dataPath="/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml")
    pun2Label = readTask3Label_2(labelPath="/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.gold")
    puns = id_puns_map.values()
    # 拼凑得到训练样本
    train_pun = []
    train_label = []
    # step1.获取所有单词
    for word,item,cur_pun in zip(pun_words,pun2Label,puns):
        synsets = wn.synsets(word) # word这个单词所有的含义
        cur_label_key = pun2Label[item]
        cur_pun = " ".join(cur_pun[0:-1])
        
        for synset in synsets:
            lemma_keys = []
            temp = wn.synset(synset.name())
            gloss = temp.definition()
            lemmas = temp.lemmas()
            for lemma in lemmas:
                lemma_key = lemma.key()
                lemma_keys.append(lemma_key)

            # 使用flag 的原因是：无论正负，只添加一个样本
            flag = False
            for lable_key in cur_label_key:
                if lable_key in lemma_keys:  
                    flag = True
        
            if flag:
                temp_text = cur_pun+" [SEP] "+gloss
                train_pun.append(temp_text)
                train_label.append(1)
            else:
                temp_text = cur_pun+" [SEP] "+gloss                                        
                train_pun.append(temp_text)
                train_label.append(0)
                        

    train_data_set = MyDataset(train_pun,train_label)
    train_data_loader = DataLoader(train_data_set,
    batch_size=args.batch_size,
    shuffle=True  # 因为生成的数据都在一块，大都比较相似，所以需要shuffle 一下
    ) #

    model = MyModel(in_fea=768,out_fea=2) # 因为只是yes/no 分类，所以这里的输出维度就是2
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(),lr = 2e-5)
    global_step = 0
    loggert_step = 50    
    for epoch in tqdm(range(args.train_epoch)):
        logger_loss = 0
        for batch in tqdm(train_data_loader):
            x,labels = batch
            labels = labels.cuda()
            inputs = tokenizer(x,max_length=args.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True)
            input_id = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()
            token_type_ids = inputs['token_type_ids'].cuda()
            
            logits = model(input_id,attention_mask,token_type_ids)
            loss = criterion(logits,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"loss={loss}")
            if global_step % loggert_step == 0 and global_step:
                viz.line([logger_loss], [global_step], win=win, update="append")
                logger_loss = 0
            global_step += 1
            logger_loss += loss.item()
        save_path = f"/home/lawson/program/punLocation/checkpoints/gloss_model_{epoch}.ckpt_1" 
        t.save(model.state_dict,save_path)


def evaluate():
    pun_words = getAllPunWords(dataPath="/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml")
    id_puns_map = getAllPuns(dataPath="/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml")
    pun2Label = readTask3Label_2(labelPath="/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.gold")
    puns = id_puns_map.values()
    # 拼凑得到验证集样本
    eval_pun = []
    eval_label = []
    # step1.获取所有单词
    for word,item,cur_pun in zip(pun_words,pun2Label,puns):
        synsets = wn.synsets(word) # word这个单词所有的含义
        cur_label_key = pun2Label[item]
        cur_pun = " ".join(cur_pun[0:-1])
        
        for synset in synsets:
            lemma_keys = []
            temp = wn.synset(synset.name())
            gloss = temp.definition()
            lemmas = temp.lemmas()
            for lemma in lemmas:
                lemma_key = lemma.key()
                lemma_keys.append(lemma_key)

            # 使用flag 的原因是：无论正负，只添加一个样本
            flag = False
            for lable_key in cur_label_key:
                if lable_key in lemma_keys:  
                    flag = True
        
            if flag:
                temp_text = cur_pun+" [SEP] "+gloss
                eval_pun.append(temp_text)
                eval_label.append(1)
            else:
                temp_text = cur_pun+" [SEP] "+gloss
                eval_pun.append(temp_text)
                eval_label.append(0)
                        
    eval_data_set = MyDataset(eval_pun,eval_label)
    eval_data_loader = DataLoader(eval_data_set,
                                  batch_size=args.batch_size,
                                  shuffle=True  # 因为生成的数据都在一块，大都比较相似，所以需要shuffle 一下
                                  )

    model = MyModel(in_fea=768,out_fea=2) # 因为只是yes/no 分类，所以这里的输出维度就是2
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    
    global_step = 0
    loggert_step = 50
    with t.no_grad():
        for epoch in tqdm(range(args.eval_epoch)):
            logger_loss = 0
            for batch in tqdm(eval_data_loader):
                x,labels = batch
                labels = labels.cuda()
                inputs = tokenizer(x,max_length=args.max_length,
                padding='max_length',
                return_tensors='pt',
                truncation=True)
                input_id = inputs['input_ids'].cuda()
                attention_mask = inputs['attention_mask'].cuda()
                token_type_ids = inputs['token_type_ids'].cuda()
                
                logits = model(input_id,attention_mask,token_type_ids)
                loss = criterion(logits,labels)                
                logger.info(f"loss={loss}")
                if global_step % loggert_step == 0 and global_step:
                    viz.line([logger_loss], [global_step], win=win, update="append")
                    logger_loss = 0
                global_step += 1                        

if __name__ == "__main__":
    print("in main")
    main()
