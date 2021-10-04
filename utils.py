import torch
import fitz
import numpy as np

from seqeval.metrics import f1_score

def split_token(token, length = None, overlap_size = None):
    """
    params sentence: list of sentences which is needed to be splitted
    params length: length of each splitted sequence
    params overlap_size: length of overlapped sequence between each 2 consecutive sequences
    return 
    """

    splitted_tokens = []
    beginning = 0
    while True:
        if beginning + length <= len(token):
            splitted_tokens.append(token[beginning: beginning + length])
        else:
            splitted_tokens.append(token[beginning:])
            break
        
        beginning = beginning + length - overlap_size
        
    return splitted_tokens

def extract_info(tokenizers, tags):
    word_list = []
    new_tags = []

    # convert token list to word list
    for token, tag in zip(tokenizers, tags):
        if tag[0] == 'X':
            word_list[-1] = word_list[-1] + token[2:]
        else:
            word_list.append(token)
            new_tags.append(tag)
    
    info = []
    value = None

    # extract info
    for word, tag in zip(word_list, new_tags):
        if tag[0] == 'B':
            value = word
            key = tag.split('-')[1]

        elif tag[0] == 'I':
            value = value + ' ' + word

        elif tag[0] == 'L':
            value = value + ' ' + word
            info.append({
                'content' : value,
                'tag' : key,
            })

        elif tag[0] == 'U':
            value = word
            key = tag.split('-')[1]
            info.append({
                'content' : value,
                'tag' : key,
            })

    return info

def extract_info_cons(tokens, tags):
    word_list = []
    new_tags = []

    # convert token list to word list
    for id in range(len(tokens)):
        if id > 0 and tokens[id - 1][-2:] == '@@':
            if (len(word_list) == 0): 
                continue
            if tokens[id][-2:] == '@@':
                word_list[-1] = word_list[-1] + tokens[id][:-2]
            else:
                word_list[-1] = word_list[-1] + tokens[id]
        else:
            if tokens[id][-2:] == '@@':
                word_list.append(tokens[id][:-2])
            else:
                word_list.append(tokens[id])
            new_tags.append(tags[id])

    info = []
    value = ''
    key = None

    # extract info
    for id in range(len(word_list)):
        if new_tags[id] == 'O' and key == None:
            value = ''

        elif (new_tags[id][0] == 'B' or new_tags[id][0] == 'U'):
            if key != None:
                info.append({
                'content' : value,
                'tag' : key,
                })
            value = word_list[id]
            key = new_tags[id].split('-')[1]

        elif (id == len(word_list) - 1 or new_tags[id][0] == 'O') and key != None:
            info.append({
                'content' : value,
                'tag' : key,
            })
            value = ''
            key = None
        elif new_tags[id][0] == 'I' and key != None:
            if new_tags[id].split('-')[1] == key:
                value = value + ' ' + word_list[id]
            else:
                info.append({
                'content' : value,
                'tag' : key,
                })
                value = ''
                key = None
         
    return info

def pdf2text(file_path):
    """
    params file_path: path to pdf file
    return text: text extracted from all pages of the pdf
    """
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.getText()

    return text

def flat_accuracy(preds, labels):
    flat_preds = np.argmax(preds, axis=2).flatten()
    flat_labels = labels.flatten()
    return np.sum(flat_preds == flat_labels)/len(flat_labels)

def evaluate(model, device, testing_loader, idx2tag):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    n_correct = 0; n_wrong = 0; total = 0
    predictions , true_labels = [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['tag'].to(device, dtype = torch.long)

            output = model(ids, attention_mask=mask, labels=targets)

            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss/nb_eval_steps
        eval_acc = eval_accuracy/nb_eval_steps
        pred_tags = [[idx2tag['LABEL_{}'.format(p_i)] for p in predictions for p_i in p]]
        valid_tags = [[idx2tag['LABEL_{}'.format(l_ii)] for l in true_labels for l_i in l for l_ii in l_i]]
        import warnings
        warnings.filterwarnings('ignore') 
        eval_f1score = f1_score(pred_tags, valid_tags)
    
    return eval_loss, eval_acc, eval_f1score

def make_prediction(
    input : str,
    idx2tag : list,
    model,
    tokenizer,
    ):

    tokens  = tokenizer.tokenize(input)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

    splitted_tokens = split_token(indexed_tokens, length=258, overlap_size=20)

    prediction = []
    for indexed_tokens in splitted_tokens:
        segments_ids = [0] * len(indexed_tokens)

        tokens_tensor = torch.tensor([indexed_tokens]).to('cpu')
        segments_tensors = torch.tensor([segments_ids]).to('cpu')
        
        model.eval()
        with torch.no_grad():
            logit = model(tokens_tensor, 
                        attention_mask=segments_tensors)

            logit_new = logit[0].argmax(2).detach().cpu().numpy().tolist()
            output = logit_new[0]
            output = [idx2tag[id] for id in output]
            tokens = tokenizer.convert_ids_to_tokens(indexed_tokens)
            result = extract_info_cons(tokens, output)
            prediction = prediction + result

    return prediction