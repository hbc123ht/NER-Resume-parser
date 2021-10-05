import torch
import fitz

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