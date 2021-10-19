import os
import json
import argparse
import torch
from transformers import PhobertTokenizer, BertConfig
import pandas as pd
from modules.model import BERTClass

from utils import pdf2text, make_prediction

if __name__ == '__main__':
    #initiate argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint_dir", dest="LOAD_CHECKPOINT_DIR", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--MAX_LEN", dest="MAX_LEN", type=int, default = 90, help="This should be strictly greater than 20")
    args = parser.parse_args()

    # load tag2idx
    with open(os.path.join(args.LOAD_CHECKPOINT_DIR, 'tag2idx.json')) as json_file:
        tag2idx = json.load(json_file)
        idx2tag = {tag2idx[key] : key for key in tag2idx.keys()}

    # initate model
    tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base", do_lower_case =False)   # initiate config
    config = BertConfig.from_pretrained("vinai/phobert-base", num_labels = len(tag2idx))

    # initial model
    model = BERTClass(config)
    model.load_state_dict(torch.load(os.path.join(args.LOAD_CHECKPOINT_DIR, 'weights.pt'),  map_location=torch.device('cpu')))

    # read data
    example = "Nguyễn Văn A Ngày sinh: 01/01/1990 Mail: abc@gmail.com 123 Hoàn Kiếm, Hà Nội  Trường Đại học Bách khoa Hà Nội"
    print(example)
    print(make_prediction(example, idx2tag, max_len = args.MAX_LEN, model = model, tokenizer = tokenizer))

