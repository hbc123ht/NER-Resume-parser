import torch
from transformers import BertModel
from .crf import CRF

class LSTMClass(torch.nn.Module):
    def __init__(self, **args):
        super(LSTMClass, self).__init__()
        self.word_embeds = torch.nn.Embedding(args['vocab_size'], args['hidden_size'])
        self.hidden_size = args['hidden_size']
        self.lstm = torch.nn.LSTM(self.hidden_size, args['hidden_size'], batch_first=True, num_layers=1)
        self.dropout = torch.nn.Dropout(args['hidden_dropout_prob'])
        self.classifier = torch.nn.Linear(args['hidden_size'], args['num_labels'])
        self.crf = CRF(num_tags = args['num_labels'], batch_first = True)
    
    def forward(self, ids, labels = None):
        embeds = self.word_embeds(ids)
        lstm_output, _ = self.lstm(embeds)
        sequence_output = self.dropout(lstm_output)
        logits = self.classifier(sequence_output)
        outputs = logits

        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels)
            outputs = -1 * loss, outputs
        return outputs