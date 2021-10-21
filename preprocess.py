import pandas as pd
import spacy
from spacy.gold import biluo_tags_from_offsets
import re

import torch

def split_sentences(sentences, length = None, overlap_size = None):
    """
    params sentence: list of sentences which is needed to be splitted
    params length: length of each splitted sequence
    params overlap_size: length of overlapped sequence between each 2 consecutive sequences
    return 
    """
    splitted_sequences = []

    for sentence in sentences:
        beginning = 0
        while True:
            if beginning + length <= len(sentence):
                splitted_sequences.append(sentence[beginning: beginning + length])
            else:
                splitted_sequences.append(sentence[beginning:])
                break
            
            beginning = beginning + length - overlap_size
        
    return splitted_sequences

def get_entities(df):
    """
    params df: annotation in "Entity recognition in resumes" dataset from kaggle 
    return entities: entities extracted from annotation with BILOU format 
    """
    entities = []
    for annos in df:
        entity = []
        annos.sort(key = lambda anno: anno['points'][0]['start'])
        for anno in annos:
            try:
                label, start, end = anno['label'][0], anno['points'][0]['start'], anno['points'][0]['end']
            except:
                continue
            
            # BILOU format
            entity.append((start, end, label))

        entities.append(entity)

    return entities

def clean_entities(contents, entities):
    """
    params contents: contents of the dataset
    params entities: entities of the dataset with BILOU format
    return fixed_entities: return entities after fixing misaligned and overlapping with BILOU format
    """

    fixed_entities = []
    for content, label in zip(contents, entities):
        fixed_entity = []

        # fix misalign
        for id, entity in enumerate(label):
            start, end = entity[0], entity[1]

            new_start, new_end = 1e9, 0

            for m in re.finditer(r'\S+', content):

                if content[start].isalnum():
                    if m.start() <= start <= m.end(): new_start = m.start()
                else:
                    if m.start() >= start: new_start = min(new_start, m.start())
                
                if content[end].isalnum():
                    if m.start() <= end <= m.end(): new_end = m.end()
                else:
                    if m.end() <= end: new_end = max(new_end, m.end())

            label[id] = (new_start, new_end, entity[2])

        #fix overlapping
        for id, entity in enumerate(label):
            
            if id < len(label) - 1 and label[id][1] >= label[id + 1][0]:
                start, end, name = label[id][0], label[id + 1][1], label[id + 1][2]
                label[id + 1] = (start, end, name)
            else:
                fixed_entity.append(label[id])

        fixed_entities.append(fixed_entity)

    return fixed_entities

def get_train_data(contents, entities):
    """
    params contents: contents of the dataset
    params entities: BILOU format entities of the dataset
    return texts, labels: list of tokens and tags
    """
    nlp = spacy.load('en_core_web_sm')
    texts = []
    labels = []

    for content, label in zip(contents, entities):
        text = []
        content = nlp(content)
        # get tokens
        for token in content:
            text.append(token.text)
        texts.append(text)

        # get tags 
        labels.append(biluo_tags_from_offsets(content, label))

    return texts, labels

def biluo_to_bio_tags(labels):
    new_labels = []

    for tags in labels:
        new_tags = []
        for tag in tags:
            if tag[0] == 'L':
                tag = tag.replace('L', 'I', 1)
            if tag[0] == 'U':
                tag = tag.replace('U', 'B', 1)
            new_tags.append(tag)

        new_labels.append(new_tags)
        
    return new_labels

def tokenize_data(contents, labels, tokenizer = None):
    tokenized_texts = []
    word_piece_labels = []

    for word_list,label in (zip(contents, labels)):
        temp_lable = []
        temp_token = []
        
        # Add [CLS] at the front 
        temp_lable.append('[CLS]')
        temp_token.append('[CLS]')
        
        for word,lab in zip(word_list,label):
            token_list = tokenizer.tokenize(word)
            for m,token in enumerate(token_list):
                temp_token.append(token)
                if m==0:
                    temp_lable.append(lab)
                else:
                    temp_lable.append('X')  
                    
        # Add [SEP] at the end
        temp_lable.append('[SEP]')
        temp_token.append('[SEP]')
        
        tokenized_texts.append(temp_token)
        word_piece_labels.append(temp_lable)
    
    return tokenized_texts, word_piece_labels

def add_sep_cls(labels):
    word_piece_labels = []
    
    for label in labels:
        temp_label = label
        temp_label.insert(0, '[CLS]')
        temp_label.append('[SEP]')
        
        word_piece_labels.append(temp_label)
    
    return word_piece_labels
