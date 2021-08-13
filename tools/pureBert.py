'''
Author: LawsonAbs
Date: 2021-01-21 09:15:36
LastEditTime: 2021-01-22 10:39:34
FilePath: /punLocation/tools/pureBert.py
'''
# 纯bert，实现一个 序列标注 任务
import torch.nn as nn
from transformers import BertModel

class  PureModel (nn.Module):
    # 因为最后输出的是单个值，所以这里的 out_fea 默认为1
    def __init__(self,path,in_fea,out_fea=1):
        super(PureModel,self).__init__()
        self.model = BertModel.from_pretrained(path)
        # 搞一个线性映射
        # 因为这里最后要预测成一个数，所以outFea 直接默认为1
        self.linear = nn.Linear(in_fea,out_fea)

    def forward(self,input_ids, token_type_ids,attention_mask):
        output = self.model(input_ids,token_type_ids, attention_mask) # 依次传入三个值
        last_hidden_layer = output[0] # 得到最后一层的输出
        
        # 取每个token的embedding，用于判断是否是双关词
        # 取每一个数据中的[CLS] 向量的值 ，用于判断是否是双关语
        token_emb = last_hidden_layer[:,1:,:] 
        # size = [batch_size,max_seq_len,768]
        cls_emb = last_hidden_layer[:,0,:]
        # size =[batch_size,1,768]
        
        out = self.linear(last_hidden_layer)  # 线性映射
        return out