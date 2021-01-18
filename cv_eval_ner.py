import pickle
import sys
from seqeval.metrics import classification_report, f1_score

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
    return classification_report(truths, preds, digits=4)

if __name__ == "__main__":
    print(main(sys.argv))

