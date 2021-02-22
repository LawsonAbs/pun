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


if __name__ == "__main__":
    # 最后运行的模块
    pathData = "/home/lawson/program/data/puns/test/homo/subtask2-homographic-test.xml"
    pathLabel = "/home/lawson/program/data/puns/test/homo/subtask2-homographic-test.gold"
    puns,punWord = simpleReadXml(pathData,pathLabel)

    wordMap = {} # 找出所有的单词 word => list
    for pun in puns:
        for word in pun: # 找出所有的单词
            word = word.lower() # 转小写            
            if word not in wordMap.keys() and word.isalpha(): # 确保是单词
                wordMap[word] = []
            else:
                continue
    print(len(wordMap)) # 有4137 个词

    defi_num = 10
    path = "./defi_emb" + str(defi_num) + ".txt"
    from transformers import BertModel,BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")
    model = BertModel.from_pretrained("/home/lawson/pretrain/bert-base-cased")

    words = list(wordMap.keys()) # 转为list
    from nltk.corpus import wordnet as wn
    import random
    # step1.获取所有单词
    for word in words:
        senses = wn.synsets(word)
        gross = []
        for _ in senses:        
            gross.append(_.definition())
        # print(gross)
            
        # step2.处理定义    
        while(len(gross) > defi_num): # 如果超过一定的设置，则随机删除某一个
            index = int(random.random() * defi_num)
            del(gross[index]) # 删除index 上的值        
            
        if len(gross)<1:
            val = word + " None\n"
            with open(path,'a') as f:
                f.write(val)
        else:    
            inputs = tokenizer(gross,
                        padding='max_length',
                        truncation=True,
                        max_length=40,
                        return_tensors='pt')
            out = model(**inputs)
            last_layer = out.last_hidden_state 
            cls_emb = last_layer[:,0,:] # 只取 CLS 上的值

            # step3.将结果写入到文件中
            cls_emb = cls_emb.tolist()
            with open(path,'a') as f: # 写入单词
                f.write(word+" ")
            for emb in cls_emb:    
                res = ""
                for data in emb:
                    #print(data)
                    res += (str(data)) 
                    res += " "
                res.strip(" ")
                res+="\n"
                with open(path,'a') as f:
                    f.write(res)