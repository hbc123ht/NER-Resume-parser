import torch
import fitz

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

        elif new_tags[id][0] == 'B' or new_tags[id][0] == 'U':
            value = word_list[id]
            key = new_tags[id].split('-')[1]

        elif (id == len(word_list) - 1 or new_tags[id][0] == 'O') and key != None:
            info.append({
                'content' : value,
                'tag' : key,
            })
            value = ''
            key = None
        elif key != None:
            value = value + ' ' + word_list[id]
         
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

def evaluate(model, idx2tag, device, valid_dataloader = None):
    model.eval()

    y_true = []
    y_pred = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=None, attention_mask=input_mask,)

        logits = logits.detach().cpu().numpy()
        logits = [list(p) for p in np.argmax(logits, axis=2)]
        
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        
        for i,mask in enumerate(input_mask):
            temp_1 = [] # Real one
            temp_2 = [] # Predict one
            
            for j, m in enumerate(mask):
                # Mark=0, meaning its a pad word, dont compare
                if m:
                    if idx2tag[label_ids[i][j]] != "X" and idx2tag[label_ids[i][j]] != "[CLS]" and idx2tag[label_ids[i][j]] != "[SEP]" : # Exclude the X label
                        temp_1.append(idx2tag[label_ids[i][j]])
                        temp_2.append(idx2tag[logits[i][j]])
                else:
                    break
            
                
            y_true.append(temp_1)
            y_pred.append(temp_2)
        
    print("f1 socre: %f"%(f1_score(y_true, y_pred)))
    print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))

    print(classification_report(y_true, y_pred,digits=4))

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