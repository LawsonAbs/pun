CUDA_VISIBLE_DEVICES=0 python cv_run_ner.py \
                        --data_dir data/semeval2017/data_with_pronunciation/pl-hete/\
                        --bert_model bert-base-cased\
                        --task_name ner \
                        --output_dir out \
                        --max_seq_length 55 \
                        --do_train \
                        --num_train_epochs 10 \
                        --pron_emb_size 16 \
                        --train_batch_size 32 \
                        --eval_batch_size 5 \
                        --do_pron \
                        --file_suffix "2021_0516_45" \
                        --use_sense \
                        --defi_num  45

                    

                    
