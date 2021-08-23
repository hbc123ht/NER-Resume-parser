import os
import numpy as np
import torch
import pandas as pd
from tqdm import trange
import argparse
import logging
import json
from utils import evaluate
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report,accuracy_score,f1_score

from transformers import BertForTokenClassification, \
                        BertTokenizer, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from preprocess import get_entities, clean_entities, tokenize_data, get_train_data


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':

    #initiate argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", dest="MAX_LEN", type=int, default=512, help="model's 'max_position_embeddings'")
    parser.add_argument("--full_finetuning", dest="FULL_FINETUNING", type=bool, default=False, help="Finetune all parameters or just only classfier parameters")
    parser.add_argument("--data_dir", dest="DATA_DIR", type=str, required=True, help="Path to DATA")
    parser.add_argument("--save_checkpoint_dir", dest="SAVE_CHECKPOINT_DIR", type=str, required=True, help="Path to save checkpoint")
    parser.add_argument("--save_checkpoint_fre", dest="SAVE_CHECKPOINT_FRE", type=int, required=True, help="num epochs per checkpoint saving")
    parser.add_argument("--num_epoch", dest="NUM_EPOCH", type=int, required=True, help="batch size")
    parser.add_argument("--batch_num", dest="BATCH_NUM", type=int, required=True, default=32, help="batch size")
    args = parser.parse_args()

    # read data
    data = pd.read_json(args.DATA_DIR, lines=True)
    data.drop('extras', axis=1, inplace=True)

    # data preprocessing
    data['entities'] = get_entities(data['annotation'])
    data['entities'] = clean_entities(data['content'], data['entities'])
    texts, labels = get_train_data(data['content'], data['entities'])

    # initiate model with BERT
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    
    # tokenize data
    tokenized_texts, word_piece_labels = tokenize_data(texts, labels, tokenizer = tokenizer)

    #create tags list
    tags_vals = list(set(i for j in word_piece_labels for i in j))
    tag2idx = {tag : idx for idx, tag in enumerate(tags_vals)}
    idx2tag = {'LABEL_{}'.format(tag2idx[key]) : key for key in tag2idx.keys()}
    with open('idx2tag.json', 'w') as outfile:
        json.dump(idx2tag, outfile)

    # Make text token into id
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                           maxlen=args.MAX_LEN, dtype="long", truncating="post", padding="post")
    
    # Make label into id, pad with "O" meaning others
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in word_piece_labels],
                      maxlen=args.MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")
        
    # For fine tune of predict, with token mask is 1,pad token is 0
    attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
    
    # Since only one sentence, all the segment set to 0
    segment_ids = [[0] * len(input_id) for input_id in input_ids]
    
    # Split all data
    tr_inputs, val_inputs, tr_tags, val_tags,tr_masks, val_masks,tr_segs, val_segs = train_test_split(input_ids, tags,attention_masks,segment_ids, 
                                                            random_state=4, test_size=0.15)

    # to torch tensor
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)
    tr_segs = torch.tensor(tr_segs)
    val_segs = torch.tensor(val_segs)

    # Only set token embedding, attention embedding, no segment embedding
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.BATCH_NUM)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.BATCH_NUM)
    
    # initiate model and set finetune method
    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels = len(tag2idx))

    model.to(device)
    model.train()

    if args.FULL_FINETUNING:
        # Fine tune model all layer parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
    else:
        # Only fine tune classifier parameters
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    # optimization method
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

    # Set epoch and grad max num
    epochs = 2
    max_grad_norm = 1.0

    # training model
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d"%(len(tr_inputs)))
    logging.info("  Batch size = %d"%(args.BATCH_NUM))

    # create checkpoint dir
    os.makedirs(args.SAVE_CHECKPOINT_DIR, exist_ok = True)

    for _ in trange(args.NUM_EPOCH,desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            # forward pass
            outputs = model(b_input_ids, token_type_ids=None,
            attention_mask=b_input_mask, labels=b_labels)
            loss, scores = outputs[:2]
            
            # backward pass
            loss.backward()
            
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            
            # update parameters
            optimizer.step()
            optimizer.zero_grad()
        
        if _ % args.SAVE_CHECKPOINT_FRE == 0 and _ > 0:
            # create folder for model
            address = os.path.join(args.SAVE_CHECKPOINT_DIR, "checkpoint_{}".format(_))
            os.makedirs(address, exist_ok = True)

            # Save a trained model, configuration and tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self  

            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(address, "pytorch_model.bin")
            output_config_file = os.path.join(address, "config.json")

            # Save model into file
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(address)

            torch.save(model.state_dict(), os.path.join('checkpoint_2', "checkpoint_{}".format(_)))

        
        # print train loss per epoch    
        logging.info("Train loss: {}".format(tr_loss/nb_tr_steps))

        # evaluate(model, idx2tag, device, valid_dataloader = valid_dataloader)

        model.eval()
