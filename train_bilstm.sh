python3 train_bilstm.py --max_len 90 \
                --data_dir 'test.json' \
                --save_checkpoint_dir 'checkpoint' \
                --save_checkpoint_fre 1 \
                --overlap_size 60 \
                --lr 3e-5 \
                --batch_num 1 \
                --num_epoch 25 \
                --pretrained_model "bert-base-multilingual-cased" \