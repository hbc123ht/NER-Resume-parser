import torch
from transformers import BertModel
from .crf import CRF

class BERTClass(torch.nn.Module):
    def __init__(self, config):
        super(BERTClass, self).__init__()
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.lstm = torch.nn.LSTM(self.hidden_size, config.hidden_size, batch_first=True, num_layers=1)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags = config.num_labels, batch_first = True)
    
    def forward(self, ids, attention_mask, labels = None):
        outputs = self.bert(ids, attention_mask, return_dict=False)
        # print(outputs)
        sequence_output = outputs[0]
        # lstm_output = sequence_output
        lstm_output, _ = self.lstm(sequence_output)
        sequence_output = self.dropout(lstm_output)
        logits = self.classifier(sequence_output)
        outputs = logits

        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs = -1 * loss, outputs
        return outputs

