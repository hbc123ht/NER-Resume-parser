python3 train.py --max_len 258 \
                --data_dir 'Vietnamese Entity Recognition in Resumes.json' \
                --save_checkpoint_dir 'checkpoint_1' \
                --save_checkpoint_fre 1 \
                --overlap_size 150 \
                --lr 3e-5 \
                --batch_num 10 \
                --num_epoch 25 \
                --pretrained_model "checkpoint_6"