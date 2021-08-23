import json
import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels = 47)
    model.load_state_dict(torch.load('checkpoint_2/checkpoint_14'))
    # model = BertForTokenClassification.from_pretrained('checkpoint_2/checkpoint_8', num_labels = 47)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    data = pd.read_json('Entity Recognition in Resumes.json', lines=True)

    example = data['content'][1]
    print(example)

    ner_results = nlp(example)
    result = []
    with open('idx2tag.json') as json_file:
        idx2tag = json.load(json_file)

    for word in ner_results:
        result.append('{}_{}'.format(word['word'], idx2tag[word['entity']]))
    
    for word in ner_results:
        print(word['word'], idx2tag[word['entity']])