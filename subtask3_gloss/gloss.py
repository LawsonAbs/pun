'''
使用pun + gloss 做二分类，然后找出最相思的两个答案
'''
import random
from sklearn.model_selection import KFold
import numpy as np
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

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--do_train",
                    action='store_true'
                    )

parser.add_argument("--do_eval",
                    action='store_true',                    
                    help="evaluate")


parser.add_argument("--train_batch_size",
                    default=16,
                    type=int,
                    required=True,
                    help="batch_size")

parser.add_argument("--train_epoch",
                    default=10,
                    type=int,
                    required=True)


parser.add_argument("--eval_batch_size",
                    default=16,
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

def setup_seed(seed):
    t.manual_seed(seed)    
    np.random.seed(seed)
    random.seed(seed)

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

"""
evaluate 数据集
"""
class EvalDataset(Dataset):
    def __init__(self,eval_puns,eval_labels):
        super(EvalDataset).__init__()
        self.puns = eval_puns
        self.labels = eval_labels
        #self.keys = eval_keys
    
    def __len__(self):
        return len(self.puns)

    def __getitem__(self, index):
        return self.puns[index],self.labels[index]#,self.keys[index]


args = parser.parse_args()

"""
计算验证的效果
"""
def cal_metric(all_pred_keys,all_true_keys):
    correct_num = 0
    for pred,true in zip(all_pred_keys,all_true_keys):
        true_left,true_right = true
        if len(pred) < 2:
            continue
        pred_left,pred_right = pred

        if true_left in pred_left:
            if true_right in pred_right:
                correct_num += 1
                continue
        
        elif true_left in pred_right:
            if true_right in pred_left:
                correct_num += 1                
                
    return correct_num / len(all_pred_keys)



def train():
    setup_seed(args.seed) 
    pun_words = getAllPunWords(dataPath="/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml")
    id_puns_map = getAllPuns(dataPath="/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml")
    pun2Label = readTask3Label_2(labelPath="/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.gold")
    # pun_words = getAllPunWords(dataPath="/home/lawson/program/punLocation/data/puns/test/homo/test.xml")
    # id_puns_map = getAllPuns(dataPath="/home/lawson/program/punLocation/data/puns/test/homo/test.xml")
    # pun2Label = readTask3Label_2(labelPath="/home/lawson/program/punLocation/data/puns/test/homo/test.gold")
    puns = list(id_puns_map.values())
    puns = np.array(puns)  # 必须转为numpy，这样才可以使用后面cv中的选择操作
    pun_words = np.array(pun_words)
    puns_id_1 = list(id_puns_map.keys())
    puns_id_2 = list(pun2Label.keys())
    pun_labels = list(pun2Label.values())
    pun_labels = np.array(pun_labels)
    pos_list = ['a','r', 'n','v'] # adj ,adv ,n ,verb
    pos_name = ['adjective','adverb','noun','verb']
    # 使用交叉验证
    kf = KFold(n_splits=10) # 分割10份
    kf.get_n_splits(puns)
    cv_index = 1
    for train_index,test_index in kf.split(puns):
        win = f"train_loss_{cv_index}"
        cv_index += 1
        raw_train_pun = puns[train_index]
        raw_train_pun_words = pun_words[train_index]
        raw_train_pun_labels = pun_labels[train_index]
        
        raw_eval_pun = puns[test_index]  
        raw_eval_pun_words = pun_words[test_index]
        raw_eval_pun_labels = pun_labels[test_index]

        # 拼凑得到训练样本
        train_pun = []
        train_label = []        
        train_key = [] # 保存每个key，用于解码时得到标签
        # step1.获取所有单词
        for word,cur_label_key,cur_pun in zip(raw_train_pun_words,raw_train_pun_labels,raw_train_pun):
            cur_pun = " ".join(cur_pun[0:-1])
            for pos,name in zip(pos_list,pos_name):
                synsets = wn.synsets(word,pos=pos) # 按pos分别找出 word这个单词所有的含义                            
                for synset in synsets:
                    lemma_keys = []
                    temp = wn.synset(synset.name())
                    gloss = temp.definition()
                    lemmas = temp.lemmas()                
                    
                    for lemma in lemmas:
                        lemma_key = lemma.key()
                        lemma_keys.append(lemma_key)            
                    train_key.append(lemma_keys) 

                    # 使用flag 的原因是：无论正负，只添加一个样本
                    flag = False
                    for lable_key in cur_label_key:
                        if lable_key in lemma_keys:  
                            flag = True
                
                    if flag:
                        temp_text = cur_pun+" [SEP] "+name+" [SEP] "+gloss
                        # temp_text = cur_pun+" [SEP] " + gloss
                        train_pun.append(temp_text)
                        train_label.append(1)
                    else:
                        temp_text = cur_pun+" [SEP] "+name+" [SEP] "+gloss
                        # temp_text = cur_pun+" [SEP] " + gloss
                        train_pun.append(temp_text)
                        train_label.append(0)
        train_pun = np.array(train_pun)
        train_label = np.array(train_label)
        train_key = np.array(train_key)
        
        
        train_data_set = MyDataset(train_pun,train_label)
        train_data_loader = DataLoader(train_data_set,
                                       batch_size=args.train_batch_size,
                                       shuffle=True  # 因为生成的数据都在一块，大都比较相似，所以需要shuffle 一下
                                       )

        model = MyModel(in_fea=768,out_fea=2) # 因为只是yes/no 分类，所以这里的输出维度就是2
        model = model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = t.optim.Adam(model.parameters(),lr = 2e-5)
        global_step = 0
        loggert_step = 50        
        max_f1 = 0
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

            # ============================================ evaluate ============================================
            logger.info("============= start evaluating =============\n")
            # 评测时，需要逐条生成结果，所以就不用batch。拼凑得到训练样本            
            # step1. 边生成，边预测
            eval_true_key = [] # 拿到eval的真实key，然后和pred计算metric 值
            all_pred_key = [] # 得到所有预测的key            
            for word,cur_label_key,cur_pun in zip(raw_eval_pun_words,raw_eval_pun_labels,raw_eval_pun):
                eval_pun = []
                eval_label = []
                cur_eval_key = [] # 保存当前这个样本的key，用于解码时得到标签
                cur_pun = " ".join(cur_pun[0:-1])
                eval_true_key.append(cur_label_key)
                for pos,name in zip(pos_list,pos_name):
                    synsets = wn.synsets(word,pos=pos) # 分词性类别（pos）的找出word这个单词所有的含义                    
                    for synset in synsets:
                        lemma_keys = []
                        temp = wn.synset(synset.name())
                        gloss = temp.definition()
                        lemmas = temp.lemmas()                
                        
                        for lemma in lemmas:
                            lemma_key = lemma.key()
                            lemma_keys.append(lemma_key)            
                        cur_eval_key.append(lemma_keys) # 这里改变成np型的数组

                        # 使用flag 的原因是：无论正负，只添加一个样本
                        flag = False
                        for lable_key in cur_label_key:
                            if lable_key in lemma_keys:  
                                flag = True
                    
                        if flag:
                            temp_text = cur_pun+" [SEP] " + name + " [SEP] " + gloss
                            # temp_text = cur_pun+" [SEP] " + gloss
                            eval_pun.append(temp_text)
                            eval_label.append(1)
                        else:
                            temp_text = cur_pun+" [SEP] " + name + " [SEP] " + gloss
                            # temp_text = cur_pun+" [SEP] " + gloss
                            eval_pun.append(temp_text)
                            eval_label.append(0)                        
                if len(eval_pun) == 0:
                    continue
                with t.no_grad():                                                             
                    inputs = tokenizer(eval_pun,
                                    max_length=args.max_length,
                                    padding='max_length',
                                    return_tensors='pt',
                                    truncation=True
                                    )
                    input_id = inputs['input_ids'].cuda()
                    attention_mask = inputs['attention_mask'].cuda()
                    token_type_ids = inputs['token_type_ids'].cuda()
                    
                    logits = model(input_id,attention_mask,token_type_ids)
                    # size = [eval_batch_size,2]
                    loss = criterion(logits,t.tensor(eval_label).cuda())
                    logger.info(f"loss={loss}")

                    # 先softmax一下，是为了更好的定阈值
                    softmax = nn.Softmax(dim=-1)
                    logits = softmax(logits)
                    pred_label = logits[:,1] # 得到预测值为1的概率
                    # threshold = 0.5                        
                    # pred_label_index = [index for index in range(len(pred_label)) if pred_label[index] > threshold]
                    # if len(pred_label_index)!=0:
                    #     pred_key = eval_key[pred_label_index]
                    #     temp = np.array(x) # 转换成numpy，方便做下面的切片操作
                    #     raw_text = temp[pred_label_index]
                    #     print(f"pred_key={pred_key},raw_text={raw_text}")
                    cnt = 0
                    pred_label_index_map = {}
                    for pred in pred_label:
                        pred_label_index_map[cnt] = pred
                        cnt += 1
                    pred_label = list(sorted(pred_label_index_map.items(),key=lambda x:x[1],reverse=True))
                    temp = []
                    # 只取top2 作为最后的释义
                    for item in pred_label[0:2]:
                        index,score = item
                        pred_key = cur_eval_key[index]
                        raw_text = eval_pun[index]
                        temp.append(pred_key)
                        logger.info(f"pred_key={pred_key},raw_text={raw_text}")
                    logger.info("\n")
                    all_pred_key.append(temp)
            
            f1 = cal_metric(all_pred_key,eval_true_key)
            max_f1 = max(f1,max_f1)
            logger.info(f"cv_index = {cv_index},epoch = {epoch},f1={f1}\n")

        logger.info(f"max_f1 = {max_f1}\n")


if __name__ == "__main__":
    
    if args.do_train:
        print("in train")
        train()    
