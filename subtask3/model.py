import torch as t
from torch.utils.data import  Dataset,DataLoader
from transformers import BertModel,BertTokenizer
import torch.nn as nn
from subtask3.attention import SelfAttention

"""
基础模型
并行的是计算。
"""
class MyModel(nn.Module):
    def __init__(self,ins,out,dim):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained("/home/lawson/pretrain/bert-base-cased")
        self.tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")
        self.attention = SelfAttention() # 搞一个attention 出来
        self.linear =nn.Linear(ins,out)
        self.softmax = nn.Softmax(dim)

    # 要使用 location 找出相关位置的向量
    # senseEmb 代表的是该双关词在wordnet 中所有含义的 emb
    def forward(self,input_ids, token_type_ids,attention_mask,location,sense_emb):
        out = self.bert(input_ids = input_ids,
                 token_type_ids = token_type_ids,
                 attention_mask=attention_mask)
        
        last_layer = out.last_hidden_state  # 取得最后一层  size = [batch_size,max_seq_length,768]
        con_pun_word_emb = []   # context pun word embedding
        # 依次从每个向量中获取双关词的向量
        for i,emb in enumerate(last_layer):
            con_pun_word_emb.append(emb[location[i],:]) # 获取双关词的emb 
        
        res = self.attention(con_pun_word_emb,sense_emb)
        res = self.linear(res)
        res = self.softmax(dim)
        res = t.argmax(res) # 返回下标 => 只能返回最大的一个
        return res # 返回最后经过softmax 之后的预测值


"""
这个类和之后的代码都是整套的。在这里，传入的data是一个字典，{input_ids:"...",tokens_type_ids:"...",attention_ask:"..."}
所以就不需要在 DataLoader()中做什么其他的操作，比如在其参数中定义一个 collate_fn=load_fn
"""
class MyDataset(Dataset):
    # 传入data【是一个字典】 和 label【是一个list】
    def __init__(self,input_ids,token_type_ids,attention_mask,location):
        super(MyDataset,self).__init__()
        # 得到三种数据
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.location = location
    
    def __len__(self) -> int:
        return len(self.input_ids)

    # 返回指定下标的训练数据和标签
    def __getitem__(self, index: int):
        return self.input_ids[index],self.token_type_ids[index],self.attention_mask[index],self.location[index]