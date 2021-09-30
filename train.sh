python3 train.py --max_len 258 \
                --data_dir 'Vietnamese Entity Recognition in Resumes.json' \
                --save_checkpoint_dir 'checkpoint_1' \
                --save_checkpoint_fre 2 \
                --batch_num 10 \
                --num_epoch 25 \
                --pretrained_model "vinai/phobert-base"