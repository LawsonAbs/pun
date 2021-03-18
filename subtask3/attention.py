import torch.nn as nn
import torch as t
import math 

"""
实现attention的操作
1.键值对attention
2.Key, Query, Value, X
"""

class SelfAttention(nn.Module):
    # 这里传入的query 是一个向量
    def __init__(self,sense_num):
        super(SelfAttention, self).__init__()        
        self.softmax = nn.Softmax(2)

        # 初始化（随机生成）三个矩阵。这三个矩阵，全局唯一
        # 其维度大小均是 (768,batch_size) 
        # 原始的self-attention 是 x * WQ = q, x * WK = k, x * WV = v。 现在更改为：
        # x * WQ = q, x * WK = k, sense_emb * WV = v
        self.WQ = t.randn(768,64,requires_grad=True).cuda()        
        self.WK = t.randn(768,64,requires_grad=True).cuda()
        self.WV = t.randn(768,64,requires_grad=True).cuda()

        self.linear = nn.Linear(64,sense_num)

    # 传入的两个参数是：输入数据X，d_k(代表的是Query 的维度0)
    """    
    x: contenxt embedding 
    sense_emb : 每个双关词的sense  embedding 集合。 size = [batch_size,sense_num,768]
    d_k: 用于缩放，初始化为64
    """
    def forward(self,x,sense_emb,d_k):
        # 求得Query , Key, Value 等值
        self.Query = t.matmul(x,self.WQ) # 向量和矩阵乘积; 计算query , matrix; size = [batch_size,64]
        self.Query = self.Query.unsqueeze(1) # size = [batch_size,1,sense_num]
        self.Key = t.matmul(sense_emb,self.WK) # 计算出key，is a matrix ; size = [sense_num,64]
        self.Key = t.transpose(self.Key,1,2) # 转置 size = [batch_size,64,sense_num]
        self.Value = t.matmul(sense_emb,self.WV) # size = [batch_size,sense_num,64]

        # 计算乘积
        out = t.matmul(self.Query,self.Key) # size = [batch_size,sense_num]
        d_k = math.sqrt(d_k)
        out = out / d_k  # sqrt(d_k)        
        score = self.softmax(out) # size = [batch_size,sense_num]
        #print(score) 
        
        res = t.matmul(score,self.Value) # size = [batch_size,64]
        res = self.linear(res) # size = [batch_size,sense_num]            
        res.squeeze_()
        return res