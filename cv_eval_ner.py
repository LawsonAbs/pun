'''
Author: LawsonAbs
Date: 2021-01-13 20:06:46
LastEditTime: 2021-01-21 19:16:22
FilePath: /punLocation/cv_eval_ner.py
'''
import pickle
import sys
from seqeval.metrics import classification_report, f1_score

def main(argv):
    file_dir = argv[1]
    #print(file_dir)
    cv = int(argv[2])
    preds,truths = [], []
    for i in range(cv):
        with open(file_dir+"pred_"+str(i), 'rb') as f:
            pred = pickle.load(f)
            preds.extend(pred)
        with open(file_dir+"true_"+str(i), 'rb') as f:
            true = pickle.load(f)
            truths.extend(true)
        assert (len(preds) == len(truths))
    return classification_report(truths, preds, digits=4)

if __name__ == "__main__":    
    '''
    简析 sys.argv  的使用：一句运行的命令中，除python 之外的都算是参数
    所以 python cv_eval_ner.py ./scores/homo-/ 10  就有三个参数，自左至右分别是argv[0] ~ argv[2]
    python cv_eval_ner.py ./scores/hete-/ 10 
    '''
    #print(len(sys.argv)) 
    print(main(sys.argv))