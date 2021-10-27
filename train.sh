python3 train.py --max_len 60 \
                --data_dir 'train.json' \
                --save_checkpoint_dir 'checkpoint' \
                --save_checkpoint_fre 1 \
                --overlap_size 20 \
                --lr 5e-5 \
                --batch_num 40 \
                --num_epoch 25 \
                --pretrained_model "vinai/phobert-base"