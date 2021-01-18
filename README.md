

# Pun Detection and Location

```
“The Boating Store Had Its Best Sail Ever”: Pronunciation-attentive Contextualized Pun Recognition
Yichao Zhou, Jyun-yu Jiang, Jieyu Zhao, Kai-Wei Chang and Wei Wang
Computer Science Department, University of California, Los Angeles
```

## Requirements

- `python3`
- `pip3 install -r requirements.txt`

## Training for Pun Detection

```Bash
python cv_run_sc.py 
--data_dir data/semeval2017/data_with_pronunciation/pd-hete/ 
--bert_model bert-base-cased 
--task_name sc 
--output_dir out 
--max_seq_length 128 
--do_train 
--num_train_epochs 3 
--do_eval 
--pron_emb_size 16 
--do_pron 
```

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
