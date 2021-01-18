import sys

def read_data(filename):

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


def main(argv):
	filename = argv[1]
	pi = argv[2]

	data = read_data(filename)

	num_tokens = 0
	num_pun_has_pi = 0
	num_pi = 0
	num_puns = 0
	num_prons = 0

	for sent in data:
	
		for unit in sent:

			label = unit[0]

			if label == P: num_puns += 1

			token = unit[1]
			prons = unit[2].split(' ')
			prons = [x.split(',') for x in prons]

			for i,tok in enumerate(tokens):
				num_tokens += 1
				if pi in prons[i]:
					num_pi += 1
				num_prons += len(prons[i] > 0)
					 

	p_y = num_pun / float(num_tokens)

if __name__ == "__main__":
	main(sys.argv)