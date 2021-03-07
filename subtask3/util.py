"""
"""
def simpleReadXml(pathData,pathLabel):
    import  xml.dom.minidom as dom
    # step1.先从pathLabel 中找出所有是双关语的句子
    allHomo = [] # 标记所有是双关词（句子中的标识）的句子集合 [hom_1,hom_2]    
    punWords = [] # 表示所有的双关词
    with open(pathLabel,'r') as f:
        line = f.readline()
        while (line!=""):
            line = line.strip() # 去行末换行
            line = line.split() # 空格分割
            #print(line[0])
            allHomo.append(line[0])            
            line = f.readline()
    #print(allHomo)
    
    # step2.接着读取双关句，成为一行文本
    #打开xml文档
    dom2 = dom.parse(pathData)
    #得到文档元素对象
    root = dom2.documentElement    
    texts = root.getElementsByTagName("text") # 得到所有的text             
    #puns = [] # 存储双关语的list
    for text in texts:        
        words = text.getElementsByTagName("word") #得到word
        pun = []
        for word in words:
            a = word.firstChild.data                
            pun.append(a)
            if word.getAttribute('id') in allHomo:
                punWords.append(a)  # 这个单词就是双关词        
    
    return punWords  # 1607条语义双关语 以及 对应的双关词

"""
根据得到的双关词列表生成含义
"""
def getSense(punWords):
    from nltk.corpus import wordnet as wn
    for word in punWords[0:1]: # word:saving
        synsets = wn.synsets(word) # synsets : [Synset('economy.n.04'), Synset('rescue.n.01'),...]
        for syn in synsets:
            # syn: Synset('economy.n.04')
            name = syn.name()  # economy.n.04
            temp = wn.synset(name) # 得到的是一个单独的Synset
            print(temp,end="\t")
            lemmas = temp.lemmas()
            print(f"{syn.definition()} =>",end="\t")
            for lemma in lemmas: # 这个for 循环中的每一项都是同义词，也就是一个意项
                print(lemma.key(),end=";")
            print("")
            


if __name__ == "__main__":
    # path = "/home/lawson/program/data/puns/test/homo/subtask3-homographic-test.gold"    
    pathData = "/home/lawson/program/data/puns/test/homo/subtask3-homographic-test.xml"
    pathLabel = "/home/lawson/program/data/puns/test/homo/subtask3-homographic-test.gold"
    punWords = simpleReadXml(pathData,pathLabel)
    print(len(punWords))    
    # for _ in punWords[0:10]:
    #     print(_)
    getSense(punWords)