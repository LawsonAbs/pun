import sys
sys.path.append(r".") # 引入当前目录作为模块，否则下面两个模块无法导入
import os
from subtask3.model import MyModel,MyDataset,EvalDataset
from subtask3.preprocess import getAllPuns, getTask3Label,getAllPunWordsSenseEmb,getPunWordEmb,readTask3Label

from transformers import BertTokenizer,BertModel
tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")
from torch.utils.data import DataLoader
import torch as t
import logging
import argparse
import torch.nn as nn
from tqdm import tqdm,trange
from sklearn.model_selection import KFold # 引入交叉验证
from visdom import Visdom # 可视化输出loss
from subtask3.util import multilabel_crossentropy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=64)
    parser.add_argument("--sense_num", # 分类的个数，这里的个数我还是觉得不应该太大。所以这里分成10个
                        type=int,
                        default=10,
                        help="the number of sense, that is the number of class")
    
    parser.add_argument("--use_random", # 填充过程使用随机数还是零填充                        
                        action='store_true',
                        help="whether to fill with random numbers")
    

    # max_seq_length 这个参数是通过计算各个pun的长度得出的结果
    parser.add_argument("--max_seq_length", 
                        type=int,
                        default=50,  # 这里的值不能太大，我觉得会影响数值
                        help="the maximum number of sequences")

    parser.add_argument("--train_epoch", 
                        type=int,
                        default=3, # 是因为经过可视化之后，发现35左右就已经收敛了
                        help="the epoch number in train")

    parser.add_argument("--expand_sample",
                        action='store_true',                        
                        help="if expand the sample") # 是否扩展样本训练

    parser.add_argument("--loss_weight",
                        default=100,
                        type = int,
                        help="help to reduce loss") # 是否扩展样本训练

    parser.add_argument("--splits_num",
                    default=5, # 因为数据量太小，所以这里就设置成5
                    type = int,
                    help="help to reduce loss") # 是否扩展样本训练

    parser.add_argument("--dropout",
                        default=0.5,
                        type=float,
                        help="drop out")
    parser.add_argument("--lr",default=2e-5,type=float,help="learning rate")

    args = parser.parse_args()

    # ============准备1. 日志相关文件============
    import time
    curTime = time.strftime("%m%d_%H%M%S", time.localtime())
    log_name = curTime + '.log'
    
    score_file = "/home/lawson/program/punLocation/subtask3/scores/"
    score_file = score_file + "homo/"+str(args.sense_num) +"_"+ str(args.train_epoch)+"/"        
    if not os.path.isdir(score_file):
        os.makedirs(score_file)

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO,
                        filename= score_file + log_name, # 以当前时间作为log名，可以指定一个文件夹
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
    #label_dict = getTask3Label(keyPath,dataPath, labelPath,outPath=None)
    
    # 这个是直接从生成好的文件中读取label_dict。 special_1.txt 中的文件去掉了 (0,1)标签
    # special_2.txt 中去掉了 (0,1) ,(0,2) 两种标签
    label_dict = readTask3Label("/home/lawson/program/punLocation/data/subtask3_labels.txt")
    sensePath = '/home/lawson/program/punLocation/data/pun_word_sense_emb.txt'
    wordEmb = getAllPunWordsSenseEmb(sensePath) # 得到双关词的所有sense 的embedding
    tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")

    # 这里将puns_dict 变成一个list的形式，all_iid 和 all_puns 是一一对应的，它们都是从 puns_dict 中抽取出来的
    # 然后将得到的这个数组，开始进行分割，执行交叉验证
    all_iid = []
    all_puns = []
    for item in puns_dict.items(): # 处理每一行 & 判断是否在label_dict 中。 这么做是为了防止出现bug，如果是 通过readTask3Label 方法所以可能
        iid, pun = item
        if iid in label_dict.keys():
            all_iid.append(iid)
            all_puns.append(pun)
        else:
            continue
    
    # 将所有参数写入到日志文件中
    logger.info(args)

    # 得到所有的数据之后，开始分割数据
    kf = KFold(n_splits=args.splits_num)
    cv_index = 1 
    for train_index, eval_index in kf.split(all_puns):
        logger.info(f"开始第 cv={cv_index} 次交叉训练")
        # 设置visdom画图的效果
        viz = Visdom()
        win = "train_loss_"+ str(cv_index)
        # ============ train中的数据需要组合生成   ==================
        # step 1.得到训练数据
        train_iid = [all_iid[i] for i in train_index]
        train_puns = [all_puns[i] for i in train_index]

        # step2. 使用上面的数据，得到在 bert model 中将使用到的项
        train_input_ids = []
        train_attention_mask = []
        train_token_type_ids = []         
        
        train_location = [] # 用于标记哪个单词是双关词
        train_labels = [] # 该双关语的一个标签
        train_total_label = [] # 该双关语的所有标签集合
        train_pun_words = []
        for i in range(len(train_puns)): # 处理每一行
            iid = train_iid[i]
            pun = train_puns[i]
            #pun = ['the', 'duke', 'and', 'the', 'count', 'had', 'a', 'fight', '.', 'the', 'duke', 'was', 'down', 'for', 'the', 'count', '.',]
            cur_total_label = label_dict[iid] # 找出当前这个双关语id对应的所有标签
            cur_location = [] # 因为要保证词被拆分完之后还能还原，所以这里用一个list 记录
            tokens = ['[CLS]'] # 当前这句话的第一个tokens
            cur_index = 0 # 用于记录 tokenizer 之后的下标
            if len(cur_total_label)==0: # 这是那种pun word 和 在key.txt 中对不上的标签
                continue
            location = (int(pun[-1])) # 最后这个位置是双关词的位置，转为int。 这个location 是从1 开始计数的
            #location = 16
            bp_word = pun[location-1] # 双关词备份   
            #bp_word = 'count'
            mask = [] # cur attention_mask
            
            raw_text_index = 1
            for word in pun[0:-1]:
                temp = tokenizer.tokenize(word)
                # 如果当前这个词就是双关词，那么就进入一个判断过程
                # 这里使用location 作为判断条件而不是 bp_word ，是因为同个双关词可能会在句中多次出现
                # 如：the duke and the count had a fight. the duke was down for the count
                if raw_text_index == location: 
                    for j in range(len(temp)):
                        cur_location.append(cur_index+j+1)
                cur_index = cur_index + len(temp)
                tokens.extend(temp) # 放入到tokens 中
                raw_text_index += 1
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

            while(len(cur_location)<5):
                cur_location.append(-1) # 以-1填充
            
            # 有几个label，就放几个到总的数据中
            # 下面这几行代码就实现了：扩充训练数据集的效果 [虽然扩充后，虽然数据集变大了，但是感觉模型不易收敛]
            for label in cur_total_label:
                temp = [0] * args.sense_num # 含义分类
                if(label[0] < args.sense_num): # 防止这个label[0] 超出了下标位置
                    temp[label[0]] = 1
                if(label[1] < args.sense_num):
                    temp[label[1]] = 1
                train_labels.append(temp) # label <class 'list'>
                train_location.append(cur_location)
                train_input_ids.append(inputs)
                train_attention_mask.append(mask)
                train_token_type_ids.append(token_type)                
                train_total_label.append(cur_total_label)
                train_pun_words.append(bp_word)
                if not args.expand_sample: # 如果不需要扩展样例，跳出for循环
                    break 
        # end-for : 处理完了所有的train data

        # 全部转为tensor 
        train_input_ids = t.tensor(train_input_ids).cuda()      
        train_token_type_ids = t.tensor(train_token_type_ids).cuda()
        train_attention_mask = t.tensor(train_attention_mask).cuda() 
        train_location = t.tensor(train_location).cuda()
        train_labels = t.tensor(train_labels, dtype=t.float).cuda()   
        #train_labels = t.tensor(train_labels, dtype=t.long).cuda()

        train_dataset = MyDataset(train_input_ids,
                            train_token_type_ids,
                            train_attention_mask,
                            train_location,
                            train_labels,
                            train_pun_words # 用于记录双关词
                            )
        
        train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True
                                )
                
        model = MyModel(args.sense_num,dropout=args.dropout)
        model.to(device)
        
        # step2. 定义模型
        # 优化器要接收模型的参数
        optimizer = t.optim.Adam(model.parameters(),lr=args.lr)
        
        logger.info("模型中需要更新的参数是：")
        # for param in model.named_parameters():
        #     print(param[0])
        
        # 使用交叉熵计算损失，因为是多标签，所以使用这个损失函数
        # 同时给这个损失函数设置一个权重值 weight
        # args.loss_weight  => negative : positive = 28:2 = 14
        pos_weight = t.full([args.sense_num],args.loss_weight).cuda()        
        #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
        #criterion = multilabel_crossentropy()

        # CrossEntropy 的参数类型： logtis 是float型， target 是long 型
        #criterion = nn.CrossEntropyLoss()  # 使用CrossEntrop
        
        # step 3. ============================ start training =====================================    
        # 这里遍历train_loader 为何会无报错跳出？ => 因为train_loader 的长度为0
        # 待生成 label 
        for epoch in trange(args.train_epoch):
            avg_loss = 0
            all_loss = 0                        
            for data in tqdm(train_dataloader):
                cur_input_ids, cur_token_type_ids, cur_attention_mask, cur_location, cur_labels,pun_words = data
                pun_words = list(pun_words) # 转为list                           

                #  根据上面找到的双关词，得到其对应的 embedding
                cur_emb = getPunWordEmb(wordEmb,pun_words,args.sense_num,args.use_random)  # 找出这个词的embedding
                # 下面这个地方填-1 的原因是：可能有时候凑不齐一个batch_size
                cur_emb = cur_emb.view(-1,args.sense_num,768)
                cur_emb = cur_emb.cuda()
                logits = model(cur_input_ids, cur_token_type_ids, cur_attention_mask,cur_location,cur_emb)  
                #logits = 10 * logits  # 简单的放缩一下
                loss = multilabel_crossentropy(logits,cur_labels)  # logits size = [batch_size,sense_num]
                all_loss += loss.item() # 累积总loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #logger.info(f"pun words in this batch are:{pun_words}")
            
            #avg_loss = all_loss/args.batch_size # 每batch的平均loss
            avg_loss = all_loss/len(train_dataloader) # 每条样本的平均loss
            # win参数代表唯一标识； update 指定了更新方式            
            viz.line([avg_loss], [epoch], win=win, update="append")
            logger.info(f"\nepoch = {epoch}, avg_loss = {avg_loss}\n")
        

        # step 4. ============================ start evaluating =====================================        
        # 从交叉验证获取 eval 数据
        eval_iid = [all_iid[i] for i in eval_index]
        eval_puns = [all_puns[i] for i in eval_index]

        eval_input_ids = []
        eval_attention_mask = []
        eval_token_type_ids = []                   
        eval_location = [] # 用于标记哪个单词是双关词
        eval_total_label = [] # 此次交叉训练中所有eval 数据的label
        eval_pun_words = [] #
        for i in range(len(eval_puns)): # 处理每一行
            iid = eval_iid[i]
            pun = eval_puns[i]
            cur_total_label = label_dict[iid] # 找出当前这个双关语id对应的所有标签
            cur_index = 0
            if len(cur_total_label)==0: # 这是那种pun word 和 在key.txt 中对不上的标签
                continue
            location = (int(pun[-1])) # 最后这个位置是双关词的位置，转为int。 这个location 是从1 开始计数的
            bp_word = pun[location-1] # 双关词备份
            eval_pun_words.append(bp_word)
            cur_location = [] # 最后这个位置是双关词的位置，转为int
            mask = [] # cur attention_mask
            tokens = ['[CLS]'] # 当前这句话的第一个tokens            

            for word in pun[0:-1]:
                temp = tokenizer.tokenize(word)
                if word== bp_word: # 如果当前这个词就是双关词                    
                    for i in range(len(temp)):
                        cur_location.append(cur_index+i+1)
                cur_index = cur_index + len(temp)
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
            
            while(len(cur_location)<5):
                cur_location.append(-1) # 以-1填充

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
                                eval_location,
                                eval_pun_words)

        eval_dataloader = DataLoader(eval_dataset, 
                                    batch_size=args.batch_size, # train  和 eval  共用同一个batch_size
                                    shuffle=False # eval 数据集就不shuffle 
                                    )


        all_pred = [] # 所有的预测结果（而不是只有一个batch），最后要放到一起计算f1 值           
        eval_pun_words = []        
        logger.info("==================== start evaluating ==============================")
        for data in tqdm(eval_dataloader):
            cur_input_ids, cur_token_type_ids, cur_attention_mask, cur_location,cur_pun_words = data                        
            cur_emb = getPunWordEmb(wordEmb,cur_pun_words,args.sense_num,args.use_random)  # 找出这个词的embedding        
            cur_emb = cur_emb.view(-1,args.sense_num,768)
            cur_emb = cur_emb.cuda()
            with t.no_grad(): # 不计算梯度
                logits = model(cur_input_ids, cur_token_type_ids, cur_attention_mask,cur_location,cur_emb)            
                # logits size = [batch_size,sense_num]
            sig = nn.Sigmoid()
            res = sig(logits) # 进行sigmoid 处理
            res = t.topk(res,k=2,dim=1) # 取top k
            index = res[1] # 取index 部分
            #index.squeeze_() # 如果是 [[2,1]] 这样的tensor，那么最后就会导致出现问题
            all_pred.extend(index.tolist())   # 放到all_pred 中
            eval_pun_words.extend(cur_pun_words)

        # TODO:将预测结果写入到文件中，以便可视化
        out_path = score_file + str(cv_index) + ".txt"
        visualizeResult(all_pred,eval_pun_words,out_path,eval_total_label)

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
        # TODO: 这个地方待寻找原因
        try:
            for label in labels:
                if pred == label:
                    right+=1
                # 因为可能pred = [6,9]，但是label是 [9,6]，所以需要将pred进行一个翻转操作，比较翻转后的数据是否相同
                # 下面这些代码 和 getTask3Label 中 的for 循环二选一
                pred.reverse()
                if pred == label:
                    right += 1
        except:
            logger.INFO("pred 的值存在问题")
    
    # 在任务3中，三者的值相同
    precision = right/len(all_pred)
    recall = right/len(all_pred)
    f1 = right/len(all_pred)
    logger.info(f"\nprecision = {precision}, recall = {recall},f1 = {f1}\n")  
    print(f"f1={f1}")


'''
可视化预测输出
1. 将预测结果写入到pred_path 中
'''
def visualizeResult(all_pred,pun_words,pred_path,all_golden):
    with open(pred_path,'w') as f:
        for i,word in enumerate(pun_words):
            pred = all_pred[i]            
            labels  =  all_golden[i] # 获取当前这个eval 的所有标签
            flag = 0                
            try:
                for label in labels:
                    if pred == label:
                        flag = 1
                    pred.reverse()
                    if pred == label:
                        flag = 1
            except:
                pass            
            pred = "("+str(pred[0] ) +","+ str(pred[1])+")" 
            gold = str(all_golden[i])
            if(flag):
                f.write("√\t"+word+"\t"+pred + ";" + gold+"\n")
            else:
                f.write("x\t"+word+"\t"+pred + ";" + gold+"\n")
            

if __name__ == "__main__":
    main()