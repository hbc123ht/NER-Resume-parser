import os
import json
import argparse
import torch
from transformers import PhobertTokenizer, BertConfig
import pandas as pd
from modules.BiLSTM_CRF import LSTMClass

from utils import pdf2text, make_prediction

if __name__ == '__main__':
    #initiate argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint_dir", dest="LOAD_CHECKPOINT_DIR", type=str, required=True, help="Path to checkpoint")
    # parser.add_argument("--load_pdf_dir", dest="LOAD_PDF_DIR", type=str, required=True, help="Path to pdf")
    args = parser.parse_args()

    # load tag2idx
    with open(os.path.join(args.LOAD_CHECKPOINT_DIR, 'tag2idx.json')) as json_file:
        tag2idx = json.load(json_file)
        idx2tag = {tag2idx[key] : key for key in tag2idx.keys()}

    # initate model
    tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base", do_lower_case =False)   

    # initiate config
    config = {
        'vocab_size' : 100000,
        'hidden_size' : 768,
        'hidden_dropout_prob' : 0.1,
        'num_labels' : len(idx2tag)
    }
    # initial model
    model = LSTMClass(**config)

    model.load_state_dict(torch.load(os.path.join(args.LOAD_CHECKPOINT_DIR, 'weights.pt'),  map_location=torch.device('cpu')))

    # read data
    data = pd.read_json('test.json', lines=True)


    for _ in range(19):
        example = data['content'][_]
        print(example)
        print(make_prediction(example, idx2tag, model = model, max_len = 90, tokenizer = tokenizer))
        print("------------------------------------------------------------------------------------------------")

