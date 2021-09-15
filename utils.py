import torch
import fitz


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

def extract_info_cons(tokenizers, tags):
    word_list = []
    new_tags = []
    
    # convert token list to word list
    for token, tag in zip(tokenizers, tags):
        if tag[0] == 'X':
            word_list[-1] = word_list[-1] + token[2:]
        else:
            word_list.append(token)
            
            if tag == 'O':
                new_tags.append(tag)
            else:
                new_tags.append(tag.split('-')[1])
    
    info = []
    value = ''
    key = None

    # extract info
    for id in range(len(word_list)):
        if new_tags[id] == 'O':
            value = ''
            continue

        if (id == len(word_list) - 1 or new_tags[id] != new_tags[id + 1]):
            value = value + ' ' + word_list[id]
            key = new_tags[id]
            info.append({
                'content' : value,
                'tag' : key,
            })
            value = ''

        else:
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