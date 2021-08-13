

# Pun Detection and Location

```
“The Boating Store Had Its Best Sail Ever”: Pronunciation-attentive Contextualized Pun Recognition
Yichao Zhou, Jyun-yu Jiang, Jieyu Zhao, Kai-Wei Chang and Wei Wang
Computer Science Department, University of California, Los Angeles
```

## Requirements

- `python3`
- `pip3 install -r requirements.txt`

## Training for Pun Location

```Bash
python cv_run_ner.py 
--data_dir data/semeval2017/data_with_pronunciation/pl-homo/ 
--bert_model bert-base-cased 
--task_name ner 
--output_dir out 
--max_seq_length 128 
--do_train 
--num_train_epochs 3 
--do_eval 
--pron_emb_size 16 
--do_pron 
```


这里简述每个文件的作用：
1. `run.sh` 是用于批量跑本代码的脚本，用于比较系统性能等。
2. `word2Sense.txt` 是用于寻找出现在训练集中的双关词有多少个含义
如:
shot	8 
就代表shot 这个单词，在训练集中有8个意思涉及到了双关意义

3. `tools` 文件夹下是工具文件
- `./tools/getSense.py`  用于生成 /home/lawson/program/punLocation/data/pun_word_definition.txt 的文件
- `./tools/generateSenseEmb.py` 是用于生成defi_emb_xxx.txt 文件
4. `defi_emb_xxx.txt` 是用于存储各个单词在 wordnet 中的不同含义
5. `pun_word_definition.txt` 是用于将每个双关词的各种含义使用bert处理得到的embedding 
6. `subtask3_v2` 是用pun+gloss 对的方式解决pun interpretation 的问题
