"""从给定的url中获取 sense2key 的字典
sense2Key 的内容格式如下：
"""
def getSenesKeyFromUrl(url,sense2Key):
    import requests # 获取请求
    from bs4 import BeautifulSoup # 引入解析包

    #url = "http://wordnetweb.princeton.edu/perl/webwn?s=sting&sub=Search+WordNet&o2=1&o0=1&o8=1&o1=1&o7=1&o5=1&o9=&o6=1&o3=1&o4=1&h=0000000000"
    html = requests.get(url)
    soup = BeautifulSoup(html.text,'lxml') #html.parser是解析器，也可是lxml
    # print(soup.prettify()) # 输出soup对象的内容
    ''' 输出内容就是：
    <html>
    <body>
    <div>
    ...
    </div>
    </body>
    </html>
    '''    
    lis = soup.find_all('li') # 从soup对象中找出所有的 li 标签
    for line in lis:   
        ''' line 中的内容如下，无换行
        <li>(2){14355490} &lt;noun.state&gt;[26] 
        <a href="webwn?o2=1&amp;o0=1&amp;o8=1&amp;o1=1&amp;o7=1&amp;o5=1&amp;o9=&amp;o6=1&amp;o3=1&amp;o4=1&amp;s=sting&amp;i=0&amp;h=0000000000#c">S:</a>
        <a class="pos"> (n) </a>
        <b>sting#1 (sting%1:26:00::)</b>, 
        <a href="webwn?o2=1&amp;o0=1&amp;o8=1&amp;o1=1&amp;o7=1&amp;o5=1&amp;o9=&amp;o6=1&amp;o3=1&amp;o4=1&amp;s=stinging">stinging#1 (stinging%1:26:00::)</a> 
        (a kind of pain; something as sudden and painful as being stung) <i>"the sting of death"; "he felt the stinging of nettles"</i></li>
        '''

        # way 3
        keys = []
        for ele in line.contents[3:-2]: # 找出其中特定下标的内容
            try:
                keys.append(ele.get_text())            
            except:
                pass
        sense = str(line.contents[-2])
        sense2Key[sense] = keys
        #print(f"{keys} => {sense}\n")  


"""
02.分析双关词的词义
"""
def simpleReadXml(pathLabel):    
    # step1.先从pathLabel 中找出所有是双关语的句子    
    allLabel = set() # 找出所有的双关词的具体释义项 [hom_1_12]    
    with open(pathLabel,'r') as f:
        line = f.readline()
        while (line!=""):
            line = line.strip() # 去行末换行
            line = line.split() # 空格分割            
            #print(line[0])
            # 将一串key分割成一个列表，然后遍历列表得到结果
            left_sense = line[1].split(";")
            right_sense = line[2].split(";")
            for sense in left_sense:
                allLabel.add(sense) # 将双关词的释义信息放入其中
            for sense in right_sense:
                allLabel.add(sense)
            line = f.readline()
     
    return allLabel  # 1607条语义双关语中的双关词

# 使用传入的word2Sense 生成一个embedding，并将这个embedding  写入到 txt 文件中
def getEmb(word2Sense,path):
    from transformers import BertModel,BertTokenizer
    model = BertModel.from_pretrained("/home/lawson/pretrain/bert-base-cased")
    tokenizer = BertTokenizer.from_pretrained("/home/lawson/pretrain/bert-base-cased")

    # 遍历所有的双关词及其含义，然后将其含义的embedding 写入到文件中
    for item in word2Sense.items():        
        key , sense_list = item
        if key == "stand":
            print("stand")
        print(f"{key}  -> {sense_list}")
        # for senses in sense_list:
        #     row = senses.split()
        #     max_sense_len = max(max_sense_len,len(row))
        inputs = tokenizer(sense_list,
                        padding='max_length',
                        truncation=True,
                        max_length=55,
                        return_tensors='pt')
        out = model(**inputs)
        last_layer = out.last_hidden_state 
        print(last_layer.size())
        cls_emb = last_layer[:,0,:] # 只取 CLS 上的值
        
        # step3.将结果写入到文件中
        # 这里生成的 sense 发现具有最多含义的双关词是 bust，有7个双关意出现在语料中。
        # 所以defi_num 参数值调整为7
        cls_emb = cls_emb.tolist()
        with open(path,'a') as f: # 写入单词
            f.write(key +" ")

        # 将 embedding 值写入到文件中 
        for emb in cls_emb:
            res = ""
            for data in emb:
                res += (str(data)) + " "                
            res.strip(" ")
            res+="\n"
            with open(path,'a') as f:   
                f.write(res)


if __name__ == "__main__":
    # 最后运行的模块
    # subtask3-homographic-test.gold
    pathLabel = "/home/lawson/program/data/puns/test/homo/subtask3-homographic-test.gold"
    #pathLabel = "/home/lawson/program/data/puns/test/homo/test.txt"
    outPath = "./pun_word_definition.txt"
    allLabel = simpleReadXml(pathLabel)
    # 内容如下： {'save%2:40:02::', 'save%2:41:02::',....}
    print(len(allLabel)) # 总共有2596个含义（可能相同）。这里的含义

    """根据wordnet找出这 2596 个词的含义，然后使用bert处理后写入到文件中
    """
    from nltk.corpus import wordnet as wn
    word2Sense = {} # word => sense 的 dict
    not_found = set()
    for label in allLabel:  # 'shake%1:13:00::'        
        #print(label)
        # step 1. 找出这个word 
        row = label.split("%")
        word = row[0]
        try:
            a = wn.lemma_from_key(label).synset()
            # print(a.definition())
            if a.definition() not in word2Sense.values() :
                if word not in word2Sense.keys(): # 如果为空
                    word2Sense[word] = [] # 先搞一个空列表                
                word2Sense[word].append(a.definition())
        except:
            not_found.add(label) 
            
    print(len(word2Sense))
    print(len(not_found)) # 有192个单词的释义在nltk 中找不到对应的解释

    with open("./statistics/word2Sense.txt",'w') as f:    
        for item in word2Sense.items():
            word, sense_list = item
            row = word +"\t"+ str(len(sense_list))+"\n"
            f.write(row)


    # 将释义使用 bert 生成 embedding 写入到文件中
    getEmb(word2Sense,outPath)

    # not_found_words = set() # 使用set 去掉重复的词
    # for _ in not_found:
    #     cur_word = _.split("%")[0]    
    #     not_found_words.add(cur_word)
    # #wn.lemma_from_key("sting%1:04:01::").synset().definition()
    # print(not_found_words)
    # print(len(not_found_words))



    # sense2Key = {} # 不同单词的sense 到 keys(也就是该单词的同义词) 的字典
    # import time
    # for word in not_found_words:
    #     url = "http://wordnetweb.princeton.edu/perl/webwn?sub=Search+WordNet&o2=1&o0=1&o8=1&o1=1&o7=1&o5=1&o9=&o6=1&o3=1&o4=1&h=0000000000"
    #     url = url + "&s=" + word
    #     getSenesKeyFromUrl(url,sense2Key)
    #     time.sleep(1) # 延迟一秒
    #     print(f"当前处理的单词是：{word}，总长度{len(sense2Key)}")


    # """
    # 从sense2Key 中找出双关词的含义
    # """
    # for word in not_found: 
    #     #print(word)
    #     for item in sense2Key.items():
    #         #print(item)
    #         key,value  = item        
    #         #print(value)
    #         for val in value:
    #             if word in val: # 如果这个单词在其中，就要保存
    #                 print(key)