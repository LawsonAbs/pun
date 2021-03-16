import sys
sys.path.append(r".") # 引入当前目录作为模块，否则下面两个模块无法导入
from subtask3.model import MyModel,MyDataset
from subtask3.preprocess import getAllPuns, getTask3Label

from transformers import BertTokenizer,BertModel
tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")
from torch.utils.data import DataLoader
import torch as t
import logging
import argparse
import torch.nn as nn
from tqdm import tqdm,trange
from sklearn.model_selection import KFold # 引入交叉验证
from sklearn.metrics import precision_score, recall_score, f1_score # 引入metric 计算
"""
01.path ：pun word sense embedding文件的路径  path = /home/lawson/program/punLocation/data/pun_word_sense_emb.txt
功能：获取 所有单词的sense embedding。
"""
def getAllPunWordsSenseEmb(path):
    wordEmb={} # {str:list}
    with open(path,'r') as f:
        line = f.readline()
        emb = [] # 装下当下单词所有的emb
        while(line):
            #print(line)
            line = line.split() # 先按照空格分割
            #if line[0][0].isalpha(): # 如果是字符，是一个新的开始
            word = line[0] # 得到单词
            if word not in wordEmb.keys():
                wordEmb[word] = []
            del(line[0]) # 删除当前的word 
            line = [float(_) for _ in line] # str -> float 
            wordEmb[word].append(line)
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
    parser.add_argument("--sense_num", # 分类的个数
                        type=int,
                        default=100,
                        help="the number of sense, that is the number of class")
    
    parser.add_argument("--use_random", # 填充过程使用随机数还是零填充                        
                        action='store_true',
                        help="whether to fill with random numbers")
    
    parser.add_argument("--max_seq_length", 
                        type=int,
                        default=100,
                        help="the maximum number of sequences")

    parser.add_argument("--train_epoch", 
                        type=int,
                        default=100,
                        help="the epoch number in train")
    args = parser.parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # step1. 定义数据
    #dataPath = '/home/lawson/program/punLocation/data/puns/test/homo/test.xml'
    dataPath = '/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml'
    puns_dict = getAllPuns(dataPath,None)

    labelPath = "/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.gold"
    keyPath = "/home/lawson/program/punLocation/data/key.txt"
    dataPath = "/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml"    
    label_dict = getTask3Label(keyPath,dataPath, labelPath,outPath=None) 
    
    sensePath = '/home/lawson/program/punLocation/data/pun_word_sense_emb.txt'
    wordEmb = getAllPunWordsSenseEmb(sensePath) # 得到双关词的sense 的embedding

    # 放到bert 中将使用的项 
    input_ids = []
    attention_mask = []
    token_type_ids = []
    tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")
    all_tokens = []
    location = [] # 用于标记哪个单词是双关词
    labels = [] # 所有数据的标签
    for item in puns_dict.items(): # 处理每一行
        iid, pun = item
        cur_label = label_dict[iid] # 找出当前这个双关语id对应的标签
        if len(cur_label)==0: # 这是那种pun word 和 在key.txt 中对不上的标签
            continue
        cur_location = int(pun[-1]) # 最后这个位置是双关词的位置，转为int
        tokens = ['[CLS]'] # 当前这句话的tokens
        mask = [] # cur attention_mask
        
        for word in pun[0:-1]:
            temp = tokenizer.tokenize(word)
            if len(temp) > 0: # 更新双关词在 tokens 序列中的位置 
                cur_location += (len(temp) - 1)
            tokens.extend(temp) # 放入到tokens 中
                    
        tokens.append("[SEP]")        

        # 如果不够长，需要分别处理三者
        mask = [1] * (len(tokens))
        
        while (len(tokens) < args.max_seq_length):
            tokens.append('[PAD]')
            mask.append(0)

         # cur input_ids
        inputs = tokenizer.convert_tokens_to_ids(tokens)
        while(len(inputs)< args.max_seq_length): 
            inputs.append(0)

        token_type = [0] * args.max_seq_length

        # 有几个label，就放几个到总的数据中
        # 下面这几行代码就实现了：扩充训练数据集的效果
        
        for label in cur_label:
            temp = [0] * args.sense_num # 含义分类
            temp[label[0]] = 1
            temp[label[1]] = 1
            labels.append(temp) # label <class 'list'>
            location.append(cur_location)
            input_ids.append(inputs)
            all_tokens.append(tokens)
            attention_mask.append(mask)
            token_type_ids.append(token_type)
    
    # 全部转为tensor形式
    input_ids = t.tensor(input_ids)
    attention_mask = t.tensor(attention_mask)
    token_type_ids = t.tensor(token_type_ids)
    labels = t.tensor(labels, dtype=t.float)
    location = t.tensor(location)

    # 得到所有的数据之后，开始分割数据
    kf = KFold(n_splits=10) # 10折交叉 
    for train_index, test_index in kf.split(input_ids): # 拿到下标
        train_index = t.tensor(train_index, dtype=t.long)
        test_index = t.tensor(test_index, dtype=t.long)

        # 获取train 的数据
        train_input_ids = t.index_select(input_ids,0,train_index).cuda()
        train_token_type_ids = t.index_select(token_type_ids,0,train_index).cuda()
        train_attention_mask = t.index_select(attention_mask,0,train_index).cuda()
        train_location = t.index_select(location,0,train_index).cuda()
        train_labels = t.index_select(labels,0,train_index).cuda()

        train_dataset = MyDataset(train_input_ids,
                            train_token_type_ids,
                            train_attention_mask,
                            train_location,
                            train_labels)
        
        train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True
                                )

        # 获取 eval 的 数据
        eval_input_ids = t.index_select(input_ids,0,test_index).cuda()
        eval_token_type_ids = t.index_select(token_type_ids,0,test_index).cuda()
        eval_attention_mask = t.index_select(attention_mask,0,test_index).cuda()
        eval_location = t.index_select(location,0,test_index).cuda()
        eval_labels = t.index_select(labels,0,test_index).cuda()

        eval_dataset = MyDataset(eval_input_ids,
                                eval_token_type_ids,
                                eval_attention_mask,
                                eval_location,
                                eval_labels)

        eval_dataloader = DataLoader(eval_dataset, 
                                    batch_size=args.batch_size, # train  和 eval  共用同一个batch_size
                                    shuffle=True
                                    )
        # 获取所有双关词的 embedding 信息
        # sense_emb = getPunWordEmb()
        model = MyModel(args.sense_num)
        model.to(device)
        
        # step2. 定义模型
        # 优化器要接收模型的参数
        optimizer = t.optim.Adam(model.parameters(),lr=1e-4)        
        criterion = nn.BCEWithLogitsLoss() # 使用交叉熵计算损失，因为是多标签，所以使用这个损失函数
        # ============================ start training =====================================    
        # 这里遍历train_loader 为何会无报错跳出？ => 因为train_loader 的长度为0
        # 待生成 label 
        for epoch in trange(args.train_epoch):
            avg_loss = 0
            all_loss = 0
            for data in tqdm(train_dataloader):
                cur_input_ids, cur_token_type_ids, cur_attention_mask, cur_location, cur_labels = data                
                pun_words = []
                # 从input_ids 中找出双关词的 embedding
                for i,input_id in enumerate(cur_input_ids):
                    iid = input_id[location[i]] # 找出这个单词的
                    word = tokenizer.convert_ids_to_tokens(iid.item())
                    pun_words.append(word)

                cur_emb = getPunWordEmb(wordEmb,pun_words,args.sense_num,args.use_random)  # 找出这个词的embedding                        
                # 下面这个地方填-1 的原因是：可能有时候凑不齐一个batch_size            
                cur_emb = cur_emb.view(-1,args.sense_num,768)
                cur_emb = cur_emb.cuda() 
                logits = model(cur_input_ids, cur_token_type_ids, cur_attention_mask,cur_location,cur_emb)                
                loss = criterion(logits,cur_labels)        
                all_loss += loss.item() # 累积总loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = all_loss/args.batch_size
            print(f"epoch = {epoch}, avg_loss in this batch is = {avg_loss}")
        
        # ============================ start evaluating =====================================        
        all_pred = [] # 所有的预测结果
        all_gold = [] # golden label
        print("==================== start evaluating ==============================")
        for data in tqdm( eval_dataloader):
            cur_input_ids, cur_token_type_ids, cur_attention_mask, cur_location, cur_labels = data 
            pun_words = []
            # 从input_ids 中找出双关词的 embedding
            for i,input_id in enumerate(cur_input_ids):
                iid = input_id[location[i]] # 找出这个单词的
                word = tokenizer.convert_ids_to_tokens(iid.item())
                pun_words.append(word)

            cur_emb = getPunWordEmb(wordEmb,pun_words,args.sense_num,args.use_random)  # 找出这个词的embedding        
            cur_emb = cur_emb.view(-1,args.sense_num,768)
            cur_emb = cur_emb.cuda()
            with t.no_grad(): # 不计算梯度
                logits = model(cur_input_ids, cur_token_type_ids, cur_attention_mask,cur_location,cur_emb)            
            # logits size = [batch_size,sense_num]
            sig = nn.Sigmoid()
            res = sig(logits) # 进行sigmoid 处理
            res = t.topk(res,k=2,dim=1) # 取top k
            index = res[1] # 取index 部分
            index.squeeze_()            
            for i in range(index.size(0)):
                pred = [0] * args.max_seq_length
                pred[index[i][0].item()] = 1
                pred[index[i][1].item()] = 1
                all_pred.extend(pred)   # 放到all_pred 中
            
            for label in cur_labels:
                all_gold.extend(label.tolist()) #        
        precision = precision_score(all_gold, all_pred, average='binary')
        recall = recall_score(all_gold, all_pred, average='binary')
        f1 = f1_score(all_gold, all_pred, average='binary')

        print(f"precision = {precision}, recall = {recall},f1 = {f1}")
        # 计算出recall, precision, f1 值

if __name__ == "__main__":
    main()