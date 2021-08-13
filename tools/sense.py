'''
Author: LawsonAbs
Date: 2021-01-20 21:50:27
LastEditTime: 2021-01-21 09:15:25
FilePath: /punLocation/tools/sense.py
'''

'''
根据sense得到各个单词的embedding，然后降到一个合适的维度。
'''

import torch as t
from nltk.corpus import wordnet as wn

# a = wn.synsets("interest")
# print(a)

# for _ in a:
#     print(_.definition())

import sys
# step1.读取文件，
if __name__ == "__main__":
    para1 = sys.argv[1]
    print(type(para1))
    print(para1)