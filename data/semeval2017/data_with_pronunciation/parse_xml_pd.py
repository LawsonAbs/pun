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

def main(argv):
    sents,ids1 = read_xml(argv[1])
    labs,ids2 = read_labels(argv[2])
    prons,ids3 = read_xml(argv[3])
    assert (ids1 == ids2)
    output = argv[4]

    train = open(output+'train.tsv','w')
    test = open(output+'test.tsv','w')
    dev  = open(output+'dev.tsv', 'w')
    for i in range(len(sents)):
        pron = ' '.join([','.join(x.split(' ')) for x in prons[i]])
        sent = str(labs[i])+'\t'+' '.join(sents[i])+'\t'+pron+'\n'
        sent = sent.encode('utf-8')
        train.write(sent)

if __name__ == "__main__":
    main(sys.argv)

