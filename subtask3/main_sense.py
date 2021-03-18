import sys
sys.path.append(r".") # 引入当前目录作为模块，否则下面两个模块无法导入
from subtask3.model import MyModel,MyDataset,EvalDataset
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

"""
功能：获取 所有单词的sense embedding
01.path ：pun word sense embedding文件的路径  path = /home/lawson/program/punLocation/data/pun_word_sense_emb.txt
"""
def getAllPunWordsSenseEmb(path):
    wordEmb={} # {str:list}
    with open(path,'r') as f:
        line = f.readline()        
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
    parser.add_argument("--sense_num", # 分类的个数，这里的个数我还是觉得不应该太大
                        type=int,
                        default=30,
                        help="the number of sense, that is the number of class")
    
    parser.add_argument("--use_random", # 填充过程使用随机数还是零填充                        
                        action='store_true',
                        help="whether to fill with random numbers")
    
    parser.add_argument("--max_seq_length", 
                        type=int,
                        default=20,  # 这里的值不能太大，我觉得会影响数值
                        help="the maximum number of sequences")

    parser.add_argument("--train_epoch", 
                        type=int,
                        default=50,
                        help="the epoch number in train")

    parser.add_argument("--expand_sample",
                        action='store_true',                        
                        help="if expand the sample") # 是否扩展样本训练

    args = parser.parse_args()

    import time
    curTime = time.strftime("%m%d_%H%M%S", time.localtime())
    log_name = curTime + '.log'
    
    score_file = "score/"
    score_file = score_file + str(args.sense_num) +"_"+ str(args.train_epoch)+"_"

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename= "/home/lawson/program/punLocation/subtask3/"+ score_file + log_name, # 以当前时间作为log名，可以指定一个文件夹
                        filemode='w', 
                        )
    logger = logging.getLogger(__name__)

    
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
    wordEmb = getAllPunWordsSenseEmb(sensePath) # 得到双关词的所有sense 的embedding
    tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")

    # 这里将puns_dict 变成一个list的形式，all_iid 和 all_puns 是一一对应的，它们都是从 puns_dict 中抽取出来的
    # 然后将得到的这个数组，开始进行分割，执行交叉验证
    all_iid = []
    all_puns = [] 
    for item in puns_dict.items(): # 处理每一行
        iid, pun = item
        all_iid.append(iid)
        all_puns.append(pun)
    
    # 得到所有的数据之后，开始分割数据
    kf = KFold(n_splits=10) # 10折交叉 
    cv_index = 1 
    for train_index, eval_index in kf.split(all_puns):
        logger.info(f"开始第 cv={cv_index} 次交叉训练")
        # ============ train中的数据需要组合生成   ==================
        # step 1.得到训练数据
        train_iid = [all_iid[i] for i in train_index]
        train_puns = [all_puns[i] for i in train_index]

        # step2. 使用上面的数据，得到在 bert model 中将使用到的项
        train_input_ids = []
        train_attention_mask = []
        train_token_type_ids = []         
        # train_all_tokens = []
        train_location = [] # 用于标记哪个单词是双关词
        train_labels = [] # 该双关语的一个标签
        train_total_label = [] # 该双关语的所有标签集合
        for i in range(len(train_puns)): # 处理每一行
            iid = train_iid[i]
            pun = train_puns[i]
            cur_total_label = label_dict[iid] # 找出当前这个双关语id对应的所有标签
            if len(cur_total_label)==0: # 这是那种pun word 和 在key.txt 中对不上的标签
                continue
            cur_location = int(pun[-1]) # 最后这个位置是双关词的位置，转为int
            tokens = ['[CLS]'] # 当前这句话的第一个tokens
            mask = [] # cur attention_mask
            
            for word in pun[0:-1]:
                temp = tokenizer.tokenize(word)
                if len(temp) > 0: # 更新双关词在 tokens 序列中的位置 
                    cur_location += (len(temp) - 1)
                tokens.extend(temp) # 放入到tokens 中
            
            tokens.append("[SEP]") # 得到一个完整的tokens
            mask = [1] * (len(tokens))
            
            # case 1.如果长度超过了最大的限制，那么取前一部分的数值，但是这样也会出现问题：因为可能会把双关词去掉，导致错误
            if(len(tokens) > args.max_seq_length):
                continue # 暂时摒弃这条样本
                # tokens = tokens[0: args.max_seq_length - 1] # 取前面一部分
                # tokens.append("[SEP]")

            # case 2.如果不够长，需要分别处理三者
            while (len(tokens) < args.max_seq_length):
                tokens.append('[PAD]')
                mask.append(0)
            
            inputs = tokenizer.convert_tokens_to_ids(tokens)
            token_type = [0] * args.max_seq_length

            
            # 有几个label，就放几个到总的数据中
            # 下面这几行代码就实现了：扩充训练数据集的效果 [虽然扩充后，虽然数据集变大了，但是感觉模型不易收敛]
            for label in cur_total_label:
                temp = [0] * args.sense_num # 含义分类
                if(label[0] < args.sense_num): # 防止这个label[0] 超出了
                    temp[label[0]] = 1
                if(label[1] < args.sense_num):
                    temp[label[1]] = 1
                train_labels.append(temp) # label <class 'list'>
                train_location.append(cur_location)
                train_input_ids.append(inputs)
                train_attention_mask.append(mask)
                train_token_type_ids.append(token_type)                
                train_total_label.append(cur_total_label)            
                if not args.expand_sample: # 如果不需要扩展样例，跳出for循环
                    break 
        # end-for : 处理完了所有的train data

        # 全部转为tensor 
        train_input_ids = t.tensor(train_input_ids).cuda()      
        train_token_type_ids = t.tensor(train_token_type_ids).cuda()
        train_attention_mask = t.tensor(train_attention_mask).cuda() 
        train_location = t.tensor(train_location).cuda()
        train_labels = t.tensor(train_labels, dtype=t.float).cuda()   

        train_dataset = MyDataset(train_input_ids,
                            train_token_type_ids,
                            train_attention_mask,
                            train_location,
                            train_labels
                            )
        
        train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True
                                )

        # 获取所有双关词的 embedding 信息
        # sense_emb = getPunWordEmb()
        model = MyModel(args.sense_num)
        model.to(device)
        
        # step2. 定义模型
        # 优化器要接收模型的参数
        optimizer = t.optim.Adam(model.parameters(),lr=1e-3)
        criterion = nn.BCEWithLogitsLoss() # 使用交叉熵计算损失，因为是多标签，所以使用这个损失函数
        
        # step 3. ============================ start training =====================================    
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
                    iid = input_id[cur_location[i]] # 找出这个单词的
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
            logger.info(f"\nepoch = {epoch}, avg_loss in this batch is {avg_loss}\n")
        

        # step 4. ============================ start evaluating =====================================        
        # 从交叉验证获取 eval 数据
        eval_iid = [all_iid[i] for i in eval_index]
        eval_puns = [all_puns[i] for i in eval_index]
                
        eval_input_ids = []
        eval_attention_mask = []
        eval_token_type_ids = []                   
        eval_location = [] # 用于标记哪个单词是双关词
        eval_total_label = [] # 此次交叉训练中所有eval 数据的label
        for i in range(len(eval_puns)): # 处理每一行
            iid = eval_iid[i]
            pun = eval_puns[i]
            cur_total_label = label_dict[iid] # 找出当前这个双关语id对应的所有标签
            if len(cur_total_label)==0: # 这是那种pun word 和 在key.txt 中对不上的标签
                continue
            cur_location = int(pun[-1]) # 最后这个位置是双关词的位置，转为int
            tokens = ['[CLS]'] # 当前这句话的第一个tokens
            mask = [] # cur attention_mask
            
            for word in pun[0:-1]:
                temp = tokenizer.tokenize(word)
                if len(temp) > 0: # 更新双关词在 tokens 序列中的位置 
                    cur_location += (len(temp) - 1)
                tokens.extend(temp) # 放入到tokens 中
            
            tokens.append("[SEP]") # 得到一个完整的tokens
            mask = [1] * (len(tokens))
            
            # case 1.如果长度超过了最大的限制，那么取前一部分的数值，但是这样也会出现问题：因为可能会把双关词去掉，导致错误
            if(len(tokens) > args.max_seq_length):
                continue # 暂时摒弃这条样本                

            # case 2.如果不够长，需要分别处理三者
            while (len(tokens) < args.max_seq_length):
                tokens.append('[PAD]')
                mask.append(0)
            
            input_id = tokenizer.convert_tokens_to_ids(tokens)
            token_type = [0] * args.max_seq_length
            
            eval_input_ids.append(input_id)
            eval_attention_mask.append(mask)
            eval_token_type_ids.append(token_type)
            eval_location.append(cur_location)
            eval_total_label.append(cur_total_label) 
        # end-for : 处理完了所有的eval data
        
        # 全部转为tensor
        eval_input_ids = t.tensor(eval_input_ids).cuda()
        eval_token_type_ids = t.tensor(eval_token_type_ids).cuda()
        eval_attention_mask = t.tensor(eval_attention_mask).cuda()
        eval_location = t.tensor(eval_location).cuda()        

        # 从特定（预划分）的数据中，获取 eval 的 数据
        eval_dataset = EvalDataset(eval_input_ids,
                                eval_token_type_ids,
                                eval_attention_mask,
                                eval_location)

        eval_dataloader = DataLoader(eval_dataset, 
                                    batch_size=args.batch_size, # train  和 eval  共用同一个batch_size
                                    shuffle=False # eval 数据集就不shuffle 
                                    )


        all_pred = [] # 所有的预测结果（而不是只有一个batch），最后要放到一起计算f1 值        
        logger.info("==================== start evaluating ==============================")
        for data in tqdm(eval_dataloader):
            cur_input_ids, cur_token_type_ids, cur_attention_mask, cur_location = data
            pun_words = []
            # 从input_ids 中找出双关词的 embedding
            for i,input_id in enumerate(cur_input_ids):
                iid = input_id[cur_location[i]] # 找出这个单词的
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
            all_pred.extend(index.tolist())   # 放到all_pred 中
        # TODO:将预测结果写入到文件中，以便可视化

        # 计算预测结果的metric
        calMetric(all_pred,eval_total_label,logger)
        cv_index+=1    


''' 计算precision， recall，f1 值
# 计算起来有点儿麻烦。因为双关语的pred_label 只有一个，而golden label可以有很多个。但是pred label 只要是 golden label 的一个子集就行了。
1.根据 input_ids 找到原来的双关语，然后找到对应的双关词，以及其golden label
2.根据得到的golden label 和 预测得到的 pred_label 计算metric
'''
def calMetric(all_pred,eval_total_label,logger):
    right = 0 # 预测正确的样例
    # 计算出recall, precision, f1 值
    for i,pred in enumerate(all_pred):
        labels  =  eval_total_label[i] # 获取当前这个eval 的所有标签
        for label in labels:
            if pred == label:
                right+=1
    
    # 在任务3中，三者的值相同
    precision = right/len(all_pred)
    recall = right/len(all_pred)
    f1 = right/len(all_pred)
    logger.info(f"\nprecision = {precision}, recall = {recall},f1 = {f1}\n")        


'''
可视化预测输出
1.
'''
def visualizeResult(all_pred,eval_input_ids,wordKeys):
    
    pass

if __name__ == "__main__":
    main()