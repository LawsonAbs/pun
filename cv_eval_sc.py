import pickle
import sys
from bert_utils import f1_2d

def main(argv):
    file_dir = argv[1]
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
    f1, recall, precision = f1_2d(truths, preds)
    return "precision, recall, f1: {}, {}, {}".format(precision*100, recall*100, f1*100)

if __name__ == "__main__":
    print(main(sys.argv))

