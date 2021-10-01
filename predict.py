import os
import json
import argparse
import torch
from transformers import PhobertTokenizer, BertForTokenClassification
import pandas as pd
from model import BERTClass

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
    model = BERTClass("vinai/phobert-base", num_labels = len(tag2idx))
    model.load_state_dict(torch.load(os.path.join(args.LOAD_CHECKPOINT_DIR, 'weights.pt')))

    # read data
    data = pd.read_json('Vietnamese Entity Recognition in Resumes.json', lines=True)

    example = data['content'][1]

    print(example)
    print(make_prediction(example, idx2tag, model = model, tokenizer = tokenizer))
