import os
import numpy as np
import torch
import pandas as pd
from tqdm import trange
import argparse
import logging
import json
from utils import evaluate

from transformers import BertConfig, PhobertTokenizer, AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import CustomDataset
from modules.model import BERTClass

from preprocess import get_entities, clean_entities, tokenize_data, get_train_data, split_sentences


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':

    #initiate argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", dest="MAX_LEN", type=int, default=512, help="model's 'max_position_embeddings'")
    parser.add_argument("--data_dir", dest="DATA_DIR", type=str, required=True, help="Path to DATA")
    parser.add_argument("--overlap_size", dest="OS", type=int, default=50, help="length of overlapped sequence between 2 sub tokens")
    parser.add_argument("--save_checkpoint_dir", dest="SAVE_CHECKPOINT_DIR", type=str, required=True, help="Path to save checkpoint")
    parser.add_argument("--save_checkpoint_fre", dest="SAVE_CHECKPOINT_FRE", type=int, required=True, help="num epochs per checkpoint saving")
    parser.add_argument("--num_epoch", dest="NUM_EPOCH", type=int, required=True, help="batch size")
    parser.add_argument("--batch_num", dest="BATCH_NUM", type=int, required=True, default=32, help="batch size")
    parser.add_argument("--lr", dest="LR", type=float, default=3e-5, help="start learning rate")
    parser.add_argument("--pretrained_model", dest="PRETRAINED_MODEL", type=str, required=True, help="Name of the pretrained model")
    parser.add_argument("--load_checkpoint_dir", dest="CHECKPOINT_DIR", type=str, default = None, help="Dir of the checkpoint")
    args = parser.parse_args()

    # read data
    data = pd.read_json(args.DATA_DIR, lines=True)

    # data preprocessing
    data['entities'] = get_entities(data['annotation'])
    data['entities'] = clean_entities(data['content'], data['entities'])
    texts, labels = get_train_data(data['content'], data['entities'])

    # initiate tokenizier
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base", do_lower_case =False)
    
    # tokenize data
    tokenized_texts, word_piece_labels = tokenize_data(texts, labels, tokenizer = tokenizer)

    tokenized_texts, word_piece_labels = split_sentences(tokenized_texts, length=args.MAX_LEN, overlap_size=args.OS), \
                    split_sentences(word_piece_labels, length=args.MAX_LEN, overlap_size=args.OS)

    train_tokenized_texts, test_tokenized_texts, \
    train_word_piece_labels, test_word_piece_labels = train_test_split(tokenized_texts, word_piece_labels, test_size=0.2, random_state=42)

    #create or load tags list

    tag2idx = None
    if args.CHECKPOINT_DIR != None:
        with open(os.path.join(args.CHECKPOINT_DIR, 'tag2idx.json')) as json_file:
            tag2idx = json.load(json_file)

    else:
        tags_vals = list(set(i for j in word_piece_labels for i in j))
        tag2idx = {tag : idx for idx, tag in enumerate(tags_vals)}

    idx2tag = {'LABEL_{}'.format(tag2idx[key]) : key for key in tag2idx.keys()}

    # initiate config
    config = BertConfig.from_pretrained(args.PRETRAINED_MODEL, num_labels = len(tag2idx))
    # initial model
    model = BERTClass(config)

    #load checkpoint if available
    if args.CHECKPOINT_DIR != None:
        model.load_state_dict(torch.load(os.path.join(args.CHECKPOINT_DIR, 'weights.pt')))

    model.to(device)
    model.train()

    # init dataset
    
    training_set = CustomDataset(train_tokenized_texts, train_word_piece_labels, tokenizer = tokenizer, max_len= args.MAX_LEN, tag2idx = tag2idx)
    test_set = CustomDataset(test_tokenized_texts, test_word_piece_labels, tokenizer = tokenizer, max_len= args.MAX_LEN, tag2idx = tag2idx)

    # Only set token embedding, attention embedding, no segment embedding
    train_params = {'batch_size': args.BATCH_NUM,
                'shuffle': True,
                'num_workers': 0
                }
    train_dataloader = DataLoader(training_set, **train_params)
    test_dataloader = DataLoader(test_set, **train_params)

    # optimization method
    optimizer = AdamW(params = model.parameters(), lr=args.LR)

    # Set epoch and grad max num
    epochs = 2
    max_grad_norm = 1.0
    torch.backends.cudnn.enabled = False
    # training model
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d"%(len(training_set)))
    logging.info("  Batch size = %d"%(args.BATCH_NUM))

    # create checkpoint dir
    os.makedirs(args.SAVE_CHECKPOINT_DIR, exist_ok = True)

    def train(epoch):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            b_input_ids = batch['ids'].to(device)
            b_input_mask = batch['mask'].to(device)
            b_labels = batch['tag'].to(device)
            
            # forward pass
            loss, output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            
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
        
        if epoch % args.SAVE_CHECKPOINT_FRE == 0 and epoch > 0:
            # create folder for model
            address = os.path.join(args.SAVE_CHECKPOINT_DIR, "checkpoint_{}".format(epoch))
            os.makedirs(address, exist_ok = True)
            torch.save(model.state_dict(), os.path.join(address, 'weights.pt'))

            # save idx2tag
            with open(os.path.join(address, 'tag2idx.json'), 'w') as outfile:
                json.dump(tag2idx, outfile)

        # evaluata
        eval_loss, eval_acc, eval_f1score = evaluate(model , testing_loader=test_dataloader, device = device, idx2tag=idx2tag)

        # print train val loss per epoch    
        logging.info("Train loss: {}".format(tr_loss/nb_tr_steps))
        logging.info("Eval loss: {}".format(eval_loss))
        logging.info("Eval accuracy: {}".format(eval_acc))
        logging.info("Eval f1 score: {}".format(eval_f1score))


    for epoch in trange(args.NUM_EPOCH,desc="Epoch"):
        train(epoch)
        