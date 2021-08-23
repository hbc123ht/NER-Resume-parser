import torch


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