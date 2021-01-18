import xml.etree.ElementTree as ET
import sys
import csv
import random

def read_xml(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    original_sentences,text_ids = [],[]

    for child in root:

        original_sentence = []
        text_id = child.attrib['id']
        for i in range(len(child)):
            original_sentence.append(child[i].text)
        original_sentences.append(original_sentence)
        text_ids.append(text_id)
    return original_sentences,text_ids

def read_labels(input_file):
    labels,label_ids = [], []

    with open(input_file, "r") as f:
        contents = f.readlines()
        for line in contents:
            vec = line.strip().split('\t')
            label_ids.append(vec[0])
            labels.append(vec[1])

    return labels,label_ids

def write_ner(sent, tag, pron, f):
    index = int(tag.split('_')[-1])
    for i in range(len(sent)):
        prons = ','.join(pron[i].split(' '))#.encode('utf-8')
        sents = sent[i]#.encode('utf-8')
        if index == i + 1:
            f.write(sents + ' ' + 'P' + ' ' + prons + '\n')
        else:
            f.write(sents + ' ' + 'O' + ' ' + prons + '\n')
    f.write('\n')

def main(argv):
    sents,ids1 = read_xml(argv[1])
    labs,ids2 = read_labels(argv[2])
    prons,ids3 = read_xml(argv[3])
    output = argv[4]

    assert (ids1 == ids2)
    assert (ids2 == ids3)

    #file_name = argv[1].replace('.xml','')
    #with open(file_name, 'w') as f:
    #    writer = csv.writer(f, delimiter='\t')
    #    for i in range(len(sents)):
    #        writer.writerow([str(labs[i]),str(' '.join(sents[i]))])

    train = open(output+'train.txt','w')
    test = open(output+'test.txt','w')
    dev  = open(output+'valid.txt', 'w')
    for i in range(len(sents)):
        print(sents[i])
        write_ner(sents[i], labs[i], prons[i], train)
        #num = random.random()
        #if num < 0.1: 
        #    write_ner(sents[i], labs[i], prons[i], dev)
        #    #dev.write(sent)
        #elif num < 0.2:
        #    write_ner(sents[i], labs[i], prons[i], test) 
        #    #test.write(sent)
        #else:
        #    write_ner(sents[i], labs[i], prons[i], train) 
        #    #train.write(sent)

if __name__ == "__main__":
    main(sys.argv)

