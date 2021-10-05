import os
import json
import torch
from transformers import PhobertTokenizer
from app.model import BERTClass
import json 

from app.utils import pdf2text, make_prediction

import yaml

with open("config.yml", "r") as stream:
    try:
        checkpoint_dir = yaml.safe_load(stream)['checkpoint_dir']
    except yaml.YAMLError as exc:
        print(exc)

# load tag2idx
with open(os.path.join(checkpoint_dir, 'tag2idx.json')) as json_file:
    tag2idx = json.load(json_file)
    idx2tag = {tag2idx[key] : key for key in tag2idx.keys()}

# initate model
tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base", do_lower_case =False)
model = BERTClass("vinai/phobert-base", num_labels = len(tag2idx))
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'weights.pt'),  map_location=torch.device('cpu')))

def resume_predict(content : str):
    output = make_prediction(content, idx2tag, model = model, tokenizer = tokenizer)
    return output