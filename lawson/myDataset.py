'''
Author: LawsonAbs
Date: 2021-01-21 19:21:45
LastEditTime: 2021-01-21 22:29:24
FilePath: /punLocation/lawson/myDataset.py
'''
from torch.utils.data import  Dataset,DataLoader


"""
这个类和之后的代码都是整套的。在这里，传入的data是一个字典，{input_ids:"...",tokens_type_ids:"...",attention_ask:"..."}
所以就不需要在 DataLoader()中做什么其他的操作，比如在其参数中定义一个 collate_fn=load_fn
"""
class MyDataset(Dataset):
    # 传入data【是一个字典】 和 label【是一个list】
    def __init__(self,data,label):
        super(MyDataset,self).__init__()
        # 得到三种数据
        self.input_ids = data.get("input_ids")
        self.token_type_ids = data.get("token_type_ids")
        self.attention_mask = data.get("attention_mask")
        
    def __len__(self) -> int:        
        return len(self.data)

    # 返回指定下标的训练数据和标签
    def __getitem__(self, index: int):
        return self.input_ids[index],self.token_type_ids[index],self.attention_mask[index],self.label[index]