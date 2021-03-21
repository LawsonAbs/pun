"""
数据预处理脚本，本脚本主要包括如下几个功能：
1.根据每个双关词生成key list。  => writeKeyAndSense
2.针对上述的双关词的每个key，为其含义生成embedding  => writeKeyAndSense
3.subtask3 生成训练数据，其目标是生成句子和标签两个文件
"""
import  xml.dom.minidom as dom
import sys
import os 
import torch as t
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer,BertModel
tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")        
model = BertModel.from_pretrained("/home/lawson/pretrain/bert-base-cased")



"""
功能：
1.对punWords 的大小写统一，否则会出现saving 和 Saving 都作为双关词的情况
pathData = subtask3-homographic-test.xml
pathLabel = subtask3-homographic-test.gold
"""
def getAllPunWords(dataPath):
    import  xml.dom.minidom as dom    
    # step2.接着读取双关句，成为一行文本
    #打开xml文档
    dom2 = dom.parse(dataPath)
    #得到文档元素对象
    root = dom2.documentElement    
    texts = root.getElementsByTagName("text") # 得到所有的text    
    punWords = [] # 表示所有的双关词
    for text in texts:
        words = text.getElementsByTagName("word") #得到word        
        for word in words:
            a = word.firstChild.data
            a = a.lower()
            if word.getAttribute('senses') == "2":
                punWords.append(a)  # 这个单词就是双关词        
    
    return punWords  # 1607条语义双关语 以及 对应的双关词

"""
功能：根据得到的双关词列表生成含义key，并将该含义生成的 embedding 写入到文件中
1.key_path 代表的是 lemma_key
2.sensePath 代表的是sense embedding 写入的文件地址
"""
def writeKeyAndSense(punWords,keyPath,sensePath):
    # step 1. 遍历所有的双关词
    for word in punWords: # word:saving
        # if word == "play_out":
        #     print("sdfsd")
        # word_2 = word
        
        synsets = wn.synsets(word) # synsets : [Synset('economy.n.04'), Synset('rescue.n.01'),...]
        # 先将双关词写入到文件中
        with open(keyPath,'a') as f:
            f.write(word+"\n")
        
        
        # step2. 找出每个双关词的所有的释义项
        # for 中的每次循环都是寻找该单词的某一种含义
        for syn in synsets:  # syn: Synset('economy.n.04')
            name = syn.name()  # economy.n.04
            temp = wn.synset(name) # 得到的是一个单独的Synset
            print(temp,end="\t")
            lemmas = temp.lemmas()
            sense = syn.definition()
            print(f"{sense} =>",end="\t")
            keys = ""
            for lemma in lemmas: # 这个for 循环中的每一项都是同义词，也就是一个义项
                print(lemma.key(),end=";")
                keys = keys+ lemma.key()+";" # keys : preservation%1:04:00::;saving%1:04:01::;
            keys = keys.strip(";") # 去行末的分号
            keys = keys + "\n"
            print("")
            with open(keyPath,'a') as f:
                f.write(keys)

            # step3.处理每个单词的sense
            # sense : an act of economizing; reduction in cost
            inputs = tokenizer(sense,return_tensors='pt')
            out = model(**inputs)
            last_layer = out.last_hidden_state 
            cls_emb = last_layer[:,0,:] # 只取 CLS 上的值

            # step4.将结果写入到文件中
            cls_emb = cls_emb.tolist()
            with open(sensePath,'a') as f: # 写入单词
                f.write(word+" ")
            for emb in cls_emb:
                res = ""
                for data in emb:
                    #print(data)
                    res += (str(data)) 
                    res += " "
                res.strip(" ")
                res+="\n"
                with open(sensePath,'a') as f:
                    f.write(res)
    


"""
得到所有的 homo 双关语。 文件内容如下：
1.outPath: 生成结果放在 subtask3_puns.txt
2.dataPath : /home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml
标签 subtask3_labels.txt
上面两个文件的每行一一对应。其中 subtask3_labels.txt 是结合 ../data/key.txt 下的文件内容生成的
"""
def getAllPuns(dataPath,outPath):
    puns = [] # 表示所有的双关词
    
    # step2.接着打开xml文档读取双关句，成为一行文本
    dom2 = dom.parse(dataPath)
    #得到文档元素对象
    root = dom2.documentElement    
    texts = root.getElementsByTagName("text") # 得到所有的text
    len_map = {} # 10:3 双关语句子长度到个数的map

    # 遍历找出每一个双关语
    for text in texts:
        id = text.getAttribute("id")  # 获取id，为了和后面的 subtask3_labels.txt 匹配
        words = text.getElementsByTagName("word") #得到word
        pun = [id]
        for word in words:
            a = word.firstChild.data
            pun.append(a)
            if word.getAttribute("senses") == '2': # 说明有两重含义
                location = word.getAttribute("id").split("_")[2]  # 下标从1 开始计数         
        # 用于计算
        cur_len = len(pun)
        if cur_len not in len_map:
            len_map[cur_len] = 1
        else:
            len_map[cur_len] += 1

        pun.append(location)
        puns.append(pun)

    # 将数据写入到文件中
    # with open(outPath,'w') as f:
    #     for pun in puns:
    #         line = ""
    #         for word in pun:
    #             line = line + word  +" "
    #         line += "\n" 
    #         f.write(line)
    #print(puns[1])

    # 对字典进行排序
    res = sorted(len_map.items(),key = lambda x:x[0])

    # 对puns 的形式做一个修改，最后以一个字典的形式返回
    # puns : 1607条语义双关语 以及 对应的双关词
    pun_dict = {} # id => pun
    for pun in puns:
        pun_dict[pun[0]] = pun[1::] 

    return pun_dict  



"""
1.根据 labelPath 和 keyPath 生成 标签数据
pun2Label = {hom_2 : [6,9];[9,6];}
keyPath = /home/lawson/program/punLocation/data/key.txt
labelPath = /home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.gold
outPath = None
"""
def getTask3Label(keyPath,dataPath,labelPath,outPath=None):    
    # step1. 先处理keyPath，生成一个排序文件
    # 这里仅仅是 economy%1:04:00:: 作为key还是不行的，因为同一个key可能会出现在多个pun word 的释义下
    # 所以需要使用 pun word + key 作为 字典中的一个key （例如，key 可以是 saving_economy%1:04:00::）
    keyMap = {} # save_economy%1:04:00:: => 0
    with open(keyPath,'r') as f:
        line = f.readline()
        cur_row = 0 # 表示当前行
        while(line): 
            if ('%' not in line): # 如果是pun word
                line = line.strip()
                word = line # pun word
                cur_row = 0
                line = f.readline()
            else:
                line = line.strip()
                line = line.split(";")            
                for key in line:
                    keyMap[word+"_"+key] = cur_row
                line = f.readline()
                cur_row += 1
    
    # step2. 从 subtask3-homographic-test.xml 读取内容，然后形成一个字典 {hom_2_9:saving...}        
    #接着打开xml文档读取双关句，成为一行文本
    dom2 = dom.parse(dataPath)
    root = dom2.documentElement    
    texts = root.getElementsByTagName("text") # 得到所有的text
    
    pun_word_dict = {} # {hom_2_9 : saving} 这种数据
    # 遍历找出每一个双关语
    for text in texts:
        id = text.getAttribute("id")  # 获取id，为了和后面的 subtask3_labels.txt 匹配
        words = text.getElementsByTagName("word") #得到word        
        for word in words:
            a = word.firstChild.data
            if word.getAttribute("senses") == '2': # 说明有两重含义
                pun_word_dict[word.getAttribute("id")] = a
    

    # step3.读取labelPath文件，然后找出所有双关词，同双关句的id形成对应
    # 例如： hom_2_9	save%2:40:02::	save%2:41:02:: 
    # 生成的结果就应该是
    pun2Label = {} # {hom_117 : [[1, 2], [3, 2], [2, 1], [2, 3]] ...}    
    with open(labelPath,'r') as f:
        line = f.readline()
        while (line):
            line = line.strip() # 去行末换行
            line = line.split() # 空格分割  
            temp = line[0] # 用以拼接temp
            
            pun_word = pun_word_dict[temp]
            temp = temp.split("_")            
            id = temp[0] + "_" + temp[1]
            # id ：hom_2249 13 
            

            # 左右两种释义
            left_sense = line[1] # grind%1:04:00::;grind%1:07:00::	
            left_sense = left_sense.split(";")
            right_sense = line[2]  # grind%1:04:01::
            right_sense = right_sense.split(";")

            # 存取左右两种位置，每次循环时重置
            left_location = set()
            right_location = set()
            
            for sense in left_sense:
                sense = pun_word + "_" + sense
                # 下面这种找不到key的共有269中情况
                if sense not in keyMap.keys():
                   # print(sense)  Todo: 待解决的问题
                    pass
                else:
                    left_location.add(keyMap[sense])
            for sense in right_sense:
                sense = pun_word + "_" + sense
                if sense not in keyMap.keys():
                    # print(sense)  Todo: 待解决的问题
                    pass
                else:
                    right_location.add(keyMap[sense])
            
            # 根据左右数据生成标签
            # 标签中的值是set的值，而不是set 中值的下标
            labels = []
            for left in (left_location):
                for right in (right_location):
                    labels.append([left,right])
            
            # for right in (right_location):
            #     for left in (left_location):
            #         labels.append([right,left])  
            line = f.readline()
            pun2Label[id] = labels
    

    # step 将标签数据写入到文件中
    # with open(outPath,'a') as f:
    #     for item in pun2Label.items():
    #         key,value = item
    #         f.write(key+"\t")
    #         for val in value:
    #             f.write(str(val) +";")    
    #         f.write("\n")
    return pun2Label
    #{hom_117 : [[1, 2], [3, 2], [2, 1], [2, 3]] ...} 



"""
功能：获取所有 双关词的sense embedding
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


if __name__ == "__main__":
    # =================================================================
    dataPath = "/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.xml"    
    outPunPath = "/home/lawson/program/punLocation/data/subtask3_puns.txt"  

    labelPath = "/home/lawson/program/punLocation/data/puns/test/homo/subtask3-homographic-test.gold"    
    outLabelPath = "/home/lawson/program/punLocation/data/subtask3_labels.txt"
    keyPath = "/home/lawson/program/punLocation/data/key.txt"
    sensePath = "/home/lawson/program/punLocation/data/pun_word_sense_emb.txt"

    if os.path.exists(outPunPath):
        os.remove(outPunPath)
    #getAllPuns(dataPath,outPunPath) # 一共有1298条数双关语
    punWords = getAllPunWords(dataPath)
    #writeKeyAndSense(punWords,keyPath,sensePath)
    
    getTask3Label(keyPath,dataPath, labelPath,outPath=outLabelPath)
