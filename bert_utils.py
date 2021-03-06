import csv
import logging
import sys
import numpy as np
import os
import random
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import json

np.random.seed(2019)

# import time
# curTime = time.strftime("%m%d_%H%M%S", time.localtime())
# log_name = curTime + '.log'
# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO,
#                     filename="./losdsg/" + log_name, # 以当前时间作为log名，可以指定一个文件夹
#                     filemode='w', # 写模式
#                     )


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, prons=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.prons = prons

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class InputPronFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, prons_id, prons_att_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.prons_id = prons_id
        self.prons_att_mask = prons_att_mask


def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename,encoding='utf-8')
    data = []
    sentence = []
    label= []
    prons = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label,prons))
                sentence = []
                label = []
                prons = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-2])
        prons.append(splits[-1][:-1].split(','))

    if len(sentence) >0:
        data.append((sentence,label,prons))
        sentence = []
        label = []
        prons = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
        return lines

"""
处理数据的主要的类
"""
class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            # 使用继承父类的读取文件的方法
            self._read_csv(os.path.join(data_dir, "train.txt")), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "valid.txt")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.txt")), "test")
    
    def get_labels(self):
        #return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        #这个具体的各个格式有什么含义？ => 类似于ner中的任务，做一个标记而已
        # 但是这里的X 是什么意思？ => bert 中的 tokenizer 过程可能会将一个词分成多个，这里词的后几部分就会被标记为X 
        #return ["O", "P", "X", "[CLS]", "[SEP]"]
        return ["O", "P", "[CLS]", "[SEP]"]

    # 创建一个训练样本
    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label,prons) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            prons = prons
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label,prons=prons))
        return examples

class ScProcessor(DataProcessor):
    """Processor for the Sentence Classification data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = None
            prons = line[2] 
            label = line[0]
            if label == "-1": label = "0"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, prons=prons, label=label))
        return examples   

# 字典里面放实例
processors = {"ner":NerProcessor,
              "sc":ScProcessor}

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    '''同文件 cv_run_ner.py 中，这里也将label_map 从0开始
    '''
    label_map = {label : i for i, label in enumerate(label_list,0)}
    
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
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
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features

# max_pron_length 表示的是发音的最长字段，如果超过了，就需要截断
# 这里是对所有的文本数据使用bert进行处理 然后组成一批输入数据
# 我在这个文件中加入了使用sense embedding 的部分
def convert_examples_to_pron_features(examples, label_list, max_seq_length, max_pron_length, tokenizer, prons_map,logger):
    """Loads a data file into a list of `InputBatch`s."""
    # 根据传入的label_list 生成了一个 label_map，也就是个字典
    label_map = {label : i for i, label in enumerate(label_list,0)}
    
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        pronslist = example.prons
        tokens = []
        labels = []
        prons = [] # 是个[[...],[...],...] 这样的数据。因为每个单词都有好几个音节，所以就需要一段（max_pron_length）来存储
        prons_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word) # 处理该词，如果改词找不到，则分块处理。
            tokens.extend(token)
            label_1 = labellist[i]
            pron_1 = pronslist[i] # the complete prons of a word
            pron_2 = [] # save the ids of prons of a word
            for j in range(len(pron_1)): 
                index = len(prons_map) # expand the map with new prons
                if pron_1[j] not in prons_map:
                    prons_map[pron_1[j]] = index + 1
                pron_2.append(prons_map[pron_1[j]])
            pron_mask_2 = [1] * len(pron_2) # 这个mask_2 的用处？
            # 对发音的字符做一个截断或者pad操作
            if len(pron_2) >= max_pron_length: 
                pron_2 = pron_2[0:max_pron_length] # trunk it if too long
                pron_mask_2 = pron_mask_2[0:max_pron_length]
            else:
                pron_2 += [0] * (max_pron_length - len(pron_2)) # pad it if too short
                pron_mask_2 += [0] * (max_pron_length - len(pron_mask_2))
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    # 也就是说，如果这个单词被切成了多段，那么发音只会被记录到第一个token分块中，后面的都用其它的填充
                    prons.append(pron_2) # only send the prons to the first piece_token of a word
                    prons_mask.append(pron_mask_2)
                else:  # 如果一个单词被 tokenize 成了两段，就会进入到这个else中。就会被标志为一个X，是在除去第一个part部分后的所有部分都会标为X
                    #labels.append("X") # 源码使用X 填充
                    labels.append("O") # 使用 O 填充
                    prons.append([0] * max_pron_length) # pad other piece_token with 0's
                    prons_mask.append([0] * max_pron_length)
            
            # 根据token 的embedding 计算相似embedding
            
        if len(tokens) >= max_seq_length - 1:  # 判断最后的sequence是否超过了最大长度 
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            prons = prons[0:(max_seq_length - 2)]
            prons_mask = prons_mask[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        prons_ids = []
        prons_att_mask = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        prons_ids.append([0] * max_pron_length) # pad the cls with 0's
        prons_att_mask.append([0] * max_pron_length)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            prons_ids.append(prons[i])
            prons_att_mask.append(prons_mask[i])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        prons_ids.append([0] * max_pron_length) # pad the sep with 0's
        prons_att_mask.append([0] * max_pron_length)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids) # 对应有token的就是1，没有就是0
        while len(input_ids) < max_seq_length: 
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            prons_ids.append([0] * max_pron_length)
            prons_att_mask.append([0] * max_pron_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(prons_ids) == max_seq_length
        assert len(prons_att_mask) == max_seq_length
        
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("prons_ids: %s" % " ".join([str(x) for x in prons_ids]))
            logger.info("prons_att_mask: %s" % " ".join([str(x) for x in prons_att_mask]))
            logger.info("prons_map: %s" % str(prons_map))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))
         
        features.append(
                InputPronFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_ids,
                                  prons_id=prons_ids,
                                  prons_att_mask=prons_att_mask))
    return features, prons_map


def convert_examples_to_pron_SC_features(examples, label_list, max_seq_length, max_pron_length, tokenizer, prons_map):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    print(label_map)
    
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        label_ids = label_map[example.label]
        pronslist = [x.split(',') for x in example.prons.split(' ')]
        #assert(len(textlist) == len(pronslist))
        if len(textlist) != len(pronslist):
            print(textlist)
            print(pronslist)
            sys.exit()
        tokens = []
        prons = []
        prons_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            pron_1 = pronslist[i] # the complete prons of a word
            pron_2 = [] # save the ids of prons of a word
            for j in range(len(pron_1)): 
                index = len(prons_map) # expand the map with new prons
                if pron_1[j] not in prons_map:
                    prons_map[pron_1[j]] = index + 1
                pron_2.append(prons_map[pron_1[j]])
            pron_mask_2 = [1] * len(pron_2)

            if len(pron_2) >= max_pron_length: 
                pron_2 = pron_2[0:max_pron_length] # trunk it if too long
                pron_mask_2 = pron_mask_2[0:max_pron_length]
            else:
                pron_2 += [0] * (max_pron_length - len(pron_2)) # pad it if too short
                pron_mask_2 += [0] * (max_pron_length - len(pron_mask_2))
            for m in range(len(token)):
                if m == 0:
                    prons.append(pron_2) # only send the prons to the first piece_token of a word
                    prons_mask.append(pron_mask_2)
                else:
                    prons.append([0] * max_pron_length) # pad other piece_token with 0's
                    prons_mask.append([0] * max_pron_length)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            prons = prons[0:(max_seq_length - 2)]
            prons_mask = prons_mask[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        prons_ids = []
        prons_att_mask = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        prons_ids.append([0] * max_pron_length) # pad the cls with 0's
        prons_att_mask.append([0] * max_pron_length)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            prons_ids.append(prons[i])
            prons_att_mask.append(prons_mask[i])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        prons_ids.append([0] * max_pron_length) # pad the sep with 0's
        prons_att_mask.append([0] * max_pron_length)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            prons_ids.append([0] * max_pron_length)
            prons_att_mask.append([0] * max_pron_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(prons_ids) == max_seq_length
        assert len(prons_att_mask) == max_seq_length
        
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("prons_ids: %s" % " ".join([str(x) for x in prons_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_ids))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputPronFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_ids,
                                  prons_id=prons_ids,
                                  prons_att_mask=prons_att_mask))
    return features, prons_map




def embed_load(file_input): # ./data/pron.16.vec  代表的是16维的发音embedding

    f = open(file_input, 'r')
    line = f.readline()
    pron_map = {}
    num,dim = line.rstrip().split(' ')# 拿到embedding 的数目和维度
    line = f.readline()
    embeddings = [[0.0]*int(dim)]
    while line != '':
        vec  = line.rstrip().split(' ')
        token = vec[0]
        emb = vec[1:]
        if token not in pron_map: 
            pron_map[token] = len(pron_map) + 1
            embeddings.append([float(x) for x in emb])  # 从str 转成float 型
    
        line = f.readline()

    return pron_map, embeddings


def embed_extend(embeddings, length):

    dim = len(embeddings[0])
    for i in range(length+1-len(embeddings)):
        embeddings.append(np.random.random([dim])*2-1)

    return embeddings

def write_scores(file_output, y):
    with open(file_output, 'wb') as f:
        pickle.dump(y, f)

def f1_2d(tmp2, tmp1):
    return f1_score(tmp2, tmp1), recall_score(tmp2,tmp1), precision_score(tmp2,tmp1)


def visualize_local(logits, label_ids, input_ids, prons_ids, prons_att_mask, att, label_map, prons_map, tokenizer):
    """
    torch.Size([8, 128])
    torch.Size([8, 128])
    torch.Size([8, 128])
    torch.Size([8, 128, 5])
    torch.Size([8, 128, 5])
    torch.Size([8, 128, 5])
    """
    prons_map = {int(prons_map[pron]): pron for pron in prons_map}

    f = open('results/pron_viz.json', 'a')
    results = {}
    for i in range(len(label_ids)):

        for j in range(len(label_ids[i])):

            ran = random.random()

            if label_ids[i][j] != 0 and label_map[label_ids[i][j]] == label_map[logits[i][j]] and label_map[label_ids[i][j]] == "P":
                mask = prons_att_mask[i][j]
                score = att[i][j]

                tmp = tokenizer.convert_ids_to_tokens(input_ids[i])
                try:
                    N = tmp.index('[PAD]')
                    results['sent'] = tmp[:N]
                except:
                    result['sent'] = tmp
                    
                results['start'] = tokenizer.convert_ids_to_tokens([int(input_ids[i][j])])[0]
                results['pron'] = {}

                for k,m in enumerate(mask):
                    if m == 0: break
                    results['pron'][prons_map[prons_ids[i][j][k]]] = float(score[k])

    json.dump(results, f)
    f.write('\n')
    return


def visualize_self(logits, label_ids, input_ids, input_mask, att, tokenizer):
    """
    torch.Size(8)
    torch.Size(8)
    torch.Size([8, 128])
    torch.Size([8, 128, 128])
    """
    f = open('results/token_viz.json', 'a')
    results = {}
    for i in range(len(input_ids)):

        if label_ids[i] == logits[i] and label_ids[i] == 1:

            try:
                N = input_mask[i].index(0)
                ids = input_ids[:N]
            except:
                ids = input_ids

            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])

            results['sent_'+str(i)] = tokens

            for j in range(len(tokens)):

                results[token +'_'+ str(j)] = att[i][j]

    json.dump(results, f)
    f.write('\n')

    return



"""
01.all_input_ids: 待转换的所有all_input_ids

"""
def getSenseEmbedding(batch_input_ids,model_dir,defi_num):
    # ================ 主要思想： 计算每个 token 对应的sense embedding  ================
        from transformers import BertTokenizerFast
        from nltk.corpus import wordnet as wn            
        sense_tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        black_list = ['is','a','be','did']
        num,max_seq_length = batch_input_ids.size()  # 得到双关语总数、每句话最大的长度
        batch_sense = [] # 存储所有单词的各个释义
        # size = [1446, max_seq_length*10]
        for input_ids in batch_input_ids: # 因为有很多句话，所以这里遍历其中的每一个双关语            
            words = sense_tokenizer.convert_ids_to_tokens(input_ids) # 将得到的token_id 转为token，一次可以转换一句话            
            for word in words[0:]: # 从头到尾判断一遍
                # 使用 wordnet 计算其sense 列表
                word = word.lower() # 转为小写
                syn = wn.synsets(word) # 获取word的含义集
                sense_list = [] # 当前这个单词的10种sense
                if word not in black_list and len(syn)>0:                    
                    for sense in syn:
                        gross = sense.definition() # 获取定义
                        sense_list.append(gross) # 追加到定义集合中
                
                # 这个参数的设置还是需要衡量一下
                # 如果单词的 sense list 不足 defi_num 个，那么就 padding 到 defi_num 个
                while(len(sense_list)<defi_num):
                    sense_list.append("")
                # 考虑随机删除某个下标是否会导致后续的出现问题
                # 如果单词的 sense list 超过 defi_num 个，那么就随机 truncate 到 defi_num 个 
                while(len(sense_list)>defi_num):
                    index = random.random() # 生成一个随机数
                    index = int(index * len(sense_list)) # 确定下标
                    del(sense_list[index]) # 删除下标为 index 的值
                batch_sense.append(sense_list) # 得到当前这句双关语的所有单词的senseList         
        
        # 处理维度
        # defi_emb = defi_emb.view(-1,defi_num,768)
        # defi_emb = defi_emb.cuda(device)
        return batch_sense # 后面交给bert处理
        # len(batch_sense) = 1446. 也就是双关语的个数
        # len(batch_sense[0]) = max_seq_length . 也就是一句双关语的最大长度
        # len(batch_sense[0][0] = defi_num) .也就是一个单词中取 defi_num 个释义

'''
 # 下面使用 bert 得到这个单词所有的 sense_list 的向量
                    senses_input_ids = sense_tokenizer(sense_list,
                                                padding='max_length',
                                                max_length=50,
                                                truncation=True,
                                                return_tensors="pt")
                    senses_output = sense_model(**senses_input_ids)
                    last_layer,other = senses_output
                    last_cls_emb = last_layer[:,0,:]  # 取任何一句话中的 CLS 向量
                    # 得到某个单词所有含义的 embdding 
                    last_cls_emb =  last_cls_emb.squeeze_(1) # 得到一个二维向量                                
                    # 拼接得到所有的结果向量
                    defi_emb = torch.cat((defi_emb,last_cls_emb),0) # 在最后一个维度上拼接即可
                else:  # 直接填充0                        
                    defi_emb = torch.cat((defi_emb,zero),0) # 在最后一个维度上拼接即可
'''

"""
01.path ：sense embedding文件的路径
功能：获取 所有单词的sense embedding
"""
def getAllWordSenseEmb(path):
    wordEmb={} # {str:list}
    with open(path,'r') as f:
        line = f.readline()    
        while(line): 
            #print(line)
            line = line.split() # 先按照空格分割
            if line[-1] == "None": # 以None结尾
                pass
            elif line[0][0].isalpha(): # 如果是字符。是一个新的开始              
                emb = [] # 装下当下单词所有的emb
                res1 = line[0] # 得到单词
                del(line[0])
                line = [float(_) for _ in line] # 全部转为float 型
                wordEmb[res1] = emb
                emb.append(line)
            else:
                line = [float(_) for _ in line] # 全部转为float 型             
                emb.append(line)
            line = f.readline()
    return wordEmb    


"""获取某句话的emb
defi_num: 表示使用的每个单词的含义个数
use_random:表示是否使用随机数填充
"""
def getPunEmb(wordEmb,words,defi_num,use_random):
    import torch as t
    t.set_printoptions(profile="full")
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

'''
将 tokens,true label,pred label 写入到txt文件中，写入的结果是：
tokens true_label pred_label
'''
def writeToTxt(tokens,true_label,pred_label,path):    
    with open(path,'w') as f:
        line = []        
        for i in range(len(tokens)): # 遍历tokens 第一维            
            for j in range(len(tokens[i])): # 遍历 tokens 第二维
                if tokens[i][j+1] == "[SEP]" :
                    break
                # line = tokens[i][j+1] +"\t"+ true_label[i][j] +"\t"+ pred_label[i][j] + "\n"
                line = f"{tokens[i][j+1]:<20}" +f"{true_label[i][j]:<3}" + f"{pred_label[i][j]:<3}" + "\n"
                f.write(line)
            f.write("\n") # 一句话写完之后，使用换行分割


if __name__ == "__main__":
    path = "./test.txt"
    # tokens = ['W','##al','-','Mart']
    # true_label = ['O','X','O','O']
    # pred_label = ['O','X','O','O']
    # writeToTxt(tokens,true_label,pred_label,path)