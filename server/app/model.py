import torch
from transformers import BertForTokenClassification

class BERTClass(torch.nn.Module):
    def __init__(self, pretrained_model : str, num_labels : int):
        super(BERTClass, self).__init__()
        self.l1 = BertForTokenClassification.from_pretrained(pretrained_model, num_labels=num_labels)
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 200)
    
    def forward(self, ids, attention_mask, labels = None):
        if labels == None:
            output_1= self.l1(ids, attention_mask)
        else:
            output_1= self.l1(ids, attention_mask, labels = labels)
        # output_2 = self.l2(output_1[0])
        # output = self.l3(output_2)
        return output_1