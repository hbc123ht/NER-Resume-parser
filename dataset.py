import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, 
                    tokens : list, 
                    tags:list, 
                    tokenizer, 
                    max_len : int, 
                    tag2idx):

        self.len = len(tokens)
        self.tokens = tokens
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx

    def __getitem__(self, index):
        token = self.tokens[index]
        ids = self.tokenizer.convert_tokens_to_ids(token)
        ids.extend([0] * (self.max_len - len(ids)))
        mask = [int(i>0) for i in ids]
        tag = self.tags[index]
        tag = [self.tag2idx.get(l) for l in tag]
        tag.extend([self.tag2idx["O"]] * (self.max_len -len(tag)))
        seg = [0] * self.max_len
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tag': torch.tensor(tag, dtype=torch.long),
            'seg': torch.tensor(seg, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len