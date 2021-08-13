'''
Author: LawsonAbs
Date: 2021-01-21 09:15:13
LastEditTime: 2021-01-22 16:43:57
FilePath: /punLocation/lawson/utils.py
'''


class InputPronFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, prons_id, prons_att_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


'''
description: 
1.读取一个xml文件，然后将得到的值写入到一个txt文件
2.去除那些不是双关语的句子
param {*} pathData
param {*} pathLabel
return {*}
res = ['They hid from the gu...at it out.', "Wal-Mart isn...ing place!', 'Can honeybee abuse l...operation?', ...]
punWord = ['sweat', 'saving', 'sting', 'entrenched', 'forge', 'hit', 'vault',...]
3.这个方法是个好方法，但是因为没有必要 将res 转换成很规范的句子。
'''
def readXml(pathData,pathLabel):
    import  xml.dom.minidom as dom
    # step1.先从pathLabel 中找出所有是双关语的句子
    allHomo = [] # 标记所有是双关语的句子集合 [hom_1,hom_2]
    allLabel =[] # 找出所有的双关词 [hom_1_12]
    punWord = [] # 表示所有的双关词
    with open(pathLabel,'r') as f:
        line = f.readline()
        while (line!=""):
            line = line.strip() # 去行末换行
            line = line.split() # 空格分割            
            #print(line[0])
            allHomo.append(line[0])
            allLabel.append(line[1]) # 将双关词的id放入其中
            line = f.readline()
    #print(allHomo)
    
    # step2.接着读取双关句，成为一行文本
    #打开xml文档
    dom2 = dom.parse(pathData)
    #得到文档元素对象
    root = dom2.documentElement    
    texts = root.getElementsByTagName("text") # 得到所有的text             
    puns = [] # 存储双关语的列表
    for text in texts:
        name = text.getAttribute('id')
        if name in allHomo:
            words = text.getElementsByTagName("word") #得到word
            pun = []
            for word in words:
                a = word.firstChild.data                
                pun.append(a)
                if word.getAttribute('id') in allLabel:
                    punWord.append(a)  # 这个单词就是双关词
            puns.append(pun)

    res = []
    sym1 = [',','.','?','!',';',':'] # 英语常用符号集合
    sym2 = ['-',':','\''] 
    for pun in puns:
        temp = ""
        for i in range(0,len(pun)):        
            cur = pun[i]
            if i!=0 :
                pre = pun[i-1]
            else: pre = "" # i为0时，pre为空
            
            if(len(cur) > 1): # 说明是单词，而不是字符
                if pre not in sym2 and pre!="": # 如果不在sym2中
                    temp=temp+ " "
                temp = temp+cur
            else: # 如果是个单个字符
                if pre in sym1:
                    temp = temp+" "
                temp = temp+cur
        res.append(temp)
    # for i in range(len(res)):
    #     print(i+1,res[i])
    return res,punWord  # 1607条语义双关语 以及 对应的双关词


'''
将数组中的值（str）写入到.txt文件中
'''
def writeToText(puns,path):
    with open(path, 'w') as f:
        for i in range(len(puns)):
            line =puns[i] +"\n" # 写入每行的内容
            f.writelines(line) # 写res 到path中
    

'''简单的方式读取文件
'''
def simpleReadXml(pathData,pathLabel):
    import  xml.dom.minidom as dom
    # step1.先从pathLabel 中找出所有是双关语的句子
    allHomo = [] # 标记所有是双关语的句子集合 [hom_1,hom_2]
    allLabel =[] # 找出所有的双关词 [hom_1_12]
    punWord = [] # 表示所有的双关词
    with open(pathLabel,'r') as f:
        line = f.readline()
        while (line!=""):
            line = line.strip() # 去行末换行
            line = line.split() # 空格分割            
            #print(line[0])
            allHomo.append(line[0])
            allLabel.append(line[1]) # 将双关词的id放入其中
            line = f.readline()
    #print(allHomo)
    
    # step2.接着读取双关句，成为一行文本
    #打开xml文档
    dom2 = dom.parse(pathData)
    #得到文档元素对象
    root = dom2.documentElement    
    texts = root.getElementsByTagName("text") # 得到所有的text             
    puns = [] # 存储双关语的列表
    for text in texts:
        name = text.getAttribute('id')
        if name in allHomo:
            words = text.getElementsByTagName("word") #得到word
            pun = []
            for word in words:
                a = word.firstChild.data                
                pun.append(a)
                if word.getAttribute('id') in allLabel:
                    punWord.append(a)  # 这个单词就是双关词
            puns.append(pun)
    
    return puns,punWord  # 1607条语义双关语 以及 对应的双关词


# 将文本数据转换成bert模型可以运行的类型
def convert_examples_to_pron_features(examples, label_list, max_seq_length,  tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    # 根据传入的label_list 生成了一个 label_map，也就是个字典
    label_map = {label : i for i, label in enumerate(label_list,1)}
    
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label        
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word) # 处理该词，如果改词找不到，则分块处理。
            tokens.extend(token)
            label_1 = labellist[i]
                                    
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)                    
                else:  # 如果一个单词被 tokenize 成了两段，就会进入到这个else中。就会被标志为一个X 
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:  # 判断最后的sequence是否超过了最大长度 
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length: 
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
         
        features.append(
                InputPronFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_ids))
    return features


"""下面写这个读取之后的接口
"""
def catEmb(wordEmb,words,defi_num):
    import torch as t
    t.set_printoptions(profile="full")
    zero = t.zeros(1,2) # 用于充当一个词的定义
    pun_sense_emb = None
    for word in words:        
        cur_word_emb = None
        if word not in wordEmb.keys(): # 根本找不到这个词。需要拼接 defi_num 次
            if cur_word_emb is None:
                cur_word_emb = zero
            while(cur_word_emb.size(0) < defi_num):
                cur_word_emb = t.cat((cur_word_emb,zero),0)            
        else:
            cur_word_emb = t.tensor(wordEmb[word])
            while (cur_word_emb.size(0) < defi_num ): # 如果小于 defi_num 个定义，则扩充到这么多
                cur_word_emb = t.cat((cur_word_emb,zero),0)
        if pun_sense_emb is None:
            pun_sense_emb = cur_word_emb
        else:
            pun_sense_emb = t.cat((pun_sense_emb,cur_word_emb),0) # 拼接得到一句话中所有的embedding
    return pun_sense_emb 


if __name__ == "__main__":
    # pathData = "/home/lawson/program/data/puns/test/homo/subtask2-homographic-test.xml"
    # pathLabel = "/home/lawson/program/data/puns/test/homo/subtask2-homographic-test.gold"
    # pun,punWord = simpleReadXml(pathData,pathLabel)
    # output_path  = "/home/lawson/program/data/puns/test/homo/2.txt"
    #writeToText(pun,output_path)
    wordEmb = {'hid':[[0.22,0.3],[0.65,0.13]],'break':[[-0.98,0.12],[0.01,0.88]]}
    puns = ["I", "hid", "from", "stock" ,"but", "I", "break","."]
    res =  catEmb(wordEmb,puns,2)
    
    print(res)