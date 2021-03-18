import torch as t
from torch.utils.data import  Dataset,DataLoader
from transformers import BertModel,BertTokenizer
import torch.nn as nn
from subtask3.attention import SelfAttention

"""
基础模型
并行的是计算。
1.是否可以将 key-value attention 衍化成多头注意力
"""
class MyModel(nn.Module):
    def __init__(self,sense_num):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained("/home/lawson/pretrain/bert-base-cased")
        self.tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")
        self.attention = SelfAttention(sense_num) # 搞一个attention 出来                
        self.softmax = nn.Softmax(1)
        self.sense_num = sense_num

    # 要使用 location 找出相关位置的向量
    # senseEmb 代表的是该双关词在wordnet 中所有含义的 emb
    def forward(self,input_ids, token_type_ids,attention_mask,location,sense_emb):
        out = self.bert(input_ids = input_ids,
                 token_type_ids = token_type_ids,
                 attention_mask=attention_mask)
        
        last_layer = out.last_hidden_state  # 取得最后一层  size = [batch_size,max_seq_length,768]
        con_pun_word_emb = None   # context pun word embedding
        # 依次从每个向量中获取双关词的向量
        for i,emb in enumerate(last_layer):
            if con_pun_word_emb is None:
                con_pun_word_emb = emb[location[i],:] # 获取双关词的emb 
            else: # 执行拼接
                con_pun_word_emb = t.cat((con_pun_word_emb,emb[location[i],:]),0)
        # con_pun_word_emb size = [batch_size,768]
        con_pun_word_emb = con_pun_word_emb.view(-1,768)
        res = self.attention(con_pun_word_emb,sense_emb,64)
        # size = [batch_size,sense_num] 
        res = res.view(-1,self.sense_num) # 修改一下形状，que
        return res 

"""
1.这个类和之后的代码都是整套的。在这里，传入的data是一个字典，{input_ids:"...",tokens_type_ids:"...",attention_ask:"..."}
所以就不需要在 DataLoader()中做什么其他的操作，比如在其参数中定义一个 collate_fn=load_fn
2.供训练部分代码使用的数据集
"""
class MyDataset(Dataset):
    # 传入data【是一个字典】 和 label【是一个list】
    def __init__(self,input_ids,token_type_ids,attention_mask,location,labels):
        super(MyDataset,self).__init__()
        # 得到三种数据
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.location = location
        self.labels = labels        
    
    def __len__(self) -> int:
        return len(self.input_ids)

    # 返回指定下标的训练数据和标签
    def __getitem__(self, index: int):
        return self.input_ids[index], self.token_type_ids[index],\
                self.attention_mask[index],self.location[index],\
                self.labels[index]


'''
1.供评测部分使用的数据集
'''
class EvalDataset(Dataset):
    # 传入data【是一个字典】 和 label【是一个list】
    def __init__(self,input_ids,token_type_ids,attention_mask,location):
        super(EvalDataset,self).__init__()
        # 得到三种数据
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.location = location           
    
    def __len__(self) -> int:
        return len(self.input_ids)

    # 返回指定下标的训练数据和标签
    def __getitem__(self, index: int):
        return self.input_ids[index], self.token_type_ids[index],self.attention_mask[index],self.location[index]                