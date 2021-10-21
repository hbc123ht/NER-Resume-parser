import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, 
                    tokens : list, 
                    tags:list, 
                    tokenizer, 
                    tag2idx):

        self.len = len(tokens)
        self.tokens = tokens
        self.tags = tags
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx

    def __getitem__(self, index):
        token = self.tokens[index]
        ids = self.tokenizer.encode(token, add_special_tokens=False)
        tag = self.tags[index]
        tag = [self.tag2idx.get(l) for l in tag]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'tag': torch.tensor(tag, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len