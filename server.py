# 1. Library imports
import uvicorn
from fastapi import FastAPI
import os
import json 
import torch
from transformers import PhobertTokenizer, BertConfig
from modules.model import BERTClass
from utils import pdf2text, make_prediction

# 2. Create app and model objects
app = FastAPI()


checkpoint_dir = os.environ["MODEL_DIR"]
MAX_LEN = int(os.environ["MAX_LEN"])
OS = int(os.environ["OS"])

# load tag2idx
with open(os.path.join(checkpoint_dir, 'tag2idx.json')) as json_file:
    tag2idx = json.load(json_file)
    idx2tag = {tag2idx[key] : key for key in tag2idx.keys()}

 # initate model
tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base", do_lower_case =False)   # initiate config
config = BertConfig.from_pretrained("vinai/phobert-base", num_labels = len(tag2idx))

# initial model
model = BERTClass(config)
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'weights.pt'),  map_location=torch.device('cpu')))

def resume_predict(content : str):
    output= make_prediction(content, idx2tag, max_len = MAX_LEN, overlap_size = OS, model = model, tokenizer = tokenizer)
    return output

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_species(content):
    data = content
    prediction = resume_predict(data)

    return prediction
