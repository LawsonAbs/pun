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
    def __init__(self):
        super(SelfAttention, self).__init__()        
        self.softmax = nn.Softmax(0)

    # 传入的两个参数是：输入数据X， d_k(代表的是Query 的维度0)
    """
    query: size = [1,768]
    key: size = [C,768] C 表示有多少类
    d_k: 用于缩放
    """
    def forward(self,query,key,value,X,d_k):
        out = t.matmul(query,key)
        d_k = math.sqrt(d_k)
        out = out * d_k  # sqrt(d_k)
        score = self.softmax(out,1)
        #print(score) 
        print(score.size()) # 得到各个的分数
        res = t.matmul(score,key)
        return res