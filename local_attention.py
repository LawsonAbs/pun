'''
Author: LawsonAbs
Date: 2021-01-13 20:06:46
LastEditTime: 2021-01-26 10:58:20
FilePath: /punLocation/local_attention.py
'''
import torch
import torch.nn as nn

"""
什么叫 Local_attention ？与global attention 相对的一种
"""
class Local_attention(nn.Module):
    # dim_in 表示的是输入矩阵的最后一个维度；dim_out 表示的是输出矩阵的最后一个维度
    def __init__(self, dim_in,dim_out):
        super(Local_attention, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear_in = nn.Linear(dim_in, dim_out, bias=False)
        self.linear_out = nn.Linear(dim_out, dim_in, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    # 这里先对数据进行mask操作
    def masked_softmax(self, T):
        T[T==0] = -10^20
        return self.softmax(T)
        
    def forward(self, context, att_vec): # att_vec size = [pron_emb_size * 2, 1]
        """
        Args:
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.  会被应用 attention 方法 的数据
                其实这个context  就是整个发音的数据
        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, pron_len, dimensions = context.size() # [4096, 5, 16]
        context = context.view(batch_size * pron_len, dimensions) # [20480,16]
        context = self.linear_in(context) 
        context = context.view(batch_size, pron_len, self.dim_out) # [4096,5,32]
        
        # 计算 attention score => 然后使用softmax 得到各个部分的权重
        attention_scores = context.matmul(att_vec) # [4096, 5, 1]
        attention_weights = self.masked_softmax(attention_scores)
        # batch mm() 
        mix = torch.bmm(attention_weights.transpose(1,2).contiguous(), context)
        mix = mix.view(batch_size, self.dim_out) # [4096,32]
        attention_weights = attention_weights.view(batch_size, pron_len) # [4096,5]

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(mix)
        output = self.tanh(output)

        return output, attention_weights # [4096,16], [4096,5]
