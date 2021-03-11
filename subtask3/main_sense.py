import sys
sys.path.append(r".") # 引入当前目录作为模块，否则下面两个模块无法导入
from subtask3.model import MyModel,MyDataset
from subtask3.preprocess import getAllPuns

from transformers import BertTokenizer,BertModel
tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")
from torch.utils.data import DataLoader
import torch as t
import logging
import argparse
import torch.nn as nn


"""
01.path ：pun word sense embedding文件的路径  path = /home/lawson/program/punLocation/data/pun_word_sense_emb.txt
功能：获取 所有单词的sense embedding。
"""
def getPunWordSenseEmb(path):
    wordEmb={} # {str:list}
    with open(path,'r') as f:
        line = f.readline()
        emb = [] # 装下当下单词所有的emb
        while(line):
            #print(line)
            line = line.split() # 先按照空格分割
            if line[0][0].isalpha(): # 如果是字符，是一个新的开始
                res1 = line[0] # 得到单词
                del(line[0])
                line = [float(_) for _ in line] # 全部转为float 型
                wordEmb[res1] = emb
                emb.append(line)
                emb = []
            else:
                line = [float(_) for _ in line] # 全部转为float 型
                emb.append(line)
            line = f.readline()
    return wordEmb



'''
从  wordEmb 中 获取某个双关词的释义，并返回。这里做了一些额外的处理：
1. 如果 word embedding 的个数不够，则需要填充
'''
def getPunWordEmb(wordEmb,words,defi_num,use_random):    
    # t.set_printoptions(profile="full") 是否打印全部的tensor 内容

    # 这里实现两种方式填充：
    # （1）使用零填充
    # （2）使用随机数填充
    if not use_random:
        pad = t.zeros(1,768) # 用于充当一个词的定义
    pun_sense_emb = None
    for word in words:
        if use_random:
            pad = t.randn(1,768)
        word = word.lower() # 转为小写
        cur_word_emb = None
        if word not in wordEmb.keys(): # 根本找不到这个词。需要拼接 defi_num 次
            if cur_word_emb is None:
                cur_word_emb = pad
            while(cur_word_emb.size(0) < defi_num):
                cur_word_emb = t.cat((cur_word_emb,pad),0)
        else:
            cur_word_emb = t.tensor(wordEmb[word])
            while (cur_word_emb.size(0) < defi_num ): # 如果小于 defi_num 个定义，则扩充到这么多
                # 在第0维拼接 0向量
                cur_word_emb = t.cat((cur_word_emb,pad),0)
            # 如果cur_word_emb.size(0) > defi_num  时需要修改
            while(cur_word_emb.size(0) > defi_num): # 只取前面的defi_num 个
                cur_word_emb = cur_word_emb[0:defi_num,:]
        if pun_sense_emb is None:
            pun_sense_emb = cur_word_emb
        else:
            pun_sense_emb = t.cat((pun_sense_emb,cur_word_emb),0) # 拼接得到一句话中所有的embedding
    return pun_sense_emb  
    # size [word_num * defi_num, defi_dim]  单词个数*含义数， 含义的维度



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=5)
    parser.add_argument("--sense_num",
                        type=int,
                        default=30,
                        help="the number of sense, that is the number of class")
    args = parser.parse_args()
    

    # step1. 定义数据
    dataPath = '/home/lawson/program/data/puns/test/homo/test.xml'
    puns = getAllPuns(dataPath,None)
    sensePath = '/home/lawson/program/punLocation/data/pun_word_sense_emb.txt'
    wordEmb = getPunWordSenseEmb(sensePath) # 得到双关词的sense 的embedding

    # 放到bert 中将使用的项 
    max_seq_length = 20
    input_ids = []
    attention_mask = []
    token_type_ids = []
    tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")
    all_tokens = []
    location = [] # 用于标记哪个单词是双关词
    for pun in puns: # 处理每一行
        cur_location = int(pun[-1]) # 最后这个位置是双关词的位置，转为int
        tokens = ['[CLS]'] # 当前这句话的tokens
        mask = [] # cur attention_mask
        
        for word in pun[1:-1]:
            temp = tokenizer.tokenize(word)
            if len(temp) > 0: # 更新双关词在 tokens 序列中的位置 
                cur_location += (len(temp) - 1)
            tokens.extend(temp) # 放入到tokens 中
            
        location.append(cur_location)
        tokens.append("[SEP]")  
        

        # 如果不够长，需要分别处理三者
        mask = [1] * (len(tokens))
        
        while (len(tokens) < max_seq_length):
            tokens.append('[PAD]')
            mask.append(0)

         # cur input_ids
        inputs = tokenizer.convert_tokens_to_ids(tokens)
        while(len(inputs)< max_seq_length): 
            inputs.append(0)

        token_type = [0] * max_seq_length

        # 放入总的数据当中
        input_ids.append(inputs)
        all_tokens.append(tokens)
        attention_mask.append(mask)        
        token_type_ids.append(token_type)
    
    # 全部转为tensor形式
    input_ids = t.tensor(input_ids)
    attention_mask = t.tensor(attention_mask)
    token_type_ids = t.tensor(token_type_ids)

    dataset = MyDataset(input_ids,
                        token_type_ids,
                        attention_mask,
                        location)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False
                            )

    # 获取所有双关词的 embedding 信息
    # sense_emb = getPunWordEmb()
    ins = 10
    out = 100
    dim = 50
    model = MyModel(ins,out,dim)

    # step2. 定义模型
    # 优化器要接收模型的参数
    optimizer = t.optim.Adam(model.parameters(),lr=1e-4)
    criterion = nn.BCELoss()  # 使用交叉熵作为损失
    
    # step3. 训练模型
    # 这里遍历dataloader 为何会无报错跳出？ => 因为dataloader 的长度为0
    # 待生成 label 
    for data in dataloader:
        input_ids, token_type_ids, attention_mask, location, labels = data
        
        sense_emb = [] # 该单词的emb
        # 从input_ids 中找出双关词的 embedding
        for input_id in enumerate(input_ids):
            iid = input_id[location] # 找出这个单词的
            punWord = tokenizer.convert_ids_to_tokens(iid)
            curEmb = wordEmb[punWord]
            sense_emb.append(curEmb)

        logits = model(input_ids, token_type_ids, attention_mask,location,sense_emb)
        loss = criterion(logits,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()