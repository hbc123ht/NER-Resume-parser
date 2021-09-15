import json
import argparse
from transformers import PhobertTokenizer, BertForTokenClassification
from transformers import pipeline
import pandas as pd
import fitz

from utils import pdf2text, extract_info, extract_info_cons

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

if __name__ == '__main__':
    #initiate argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint_dir", dest="LOAD_CHECKPOINT_DIR", type=str, required=True, help="Path to checkpoint")
    # parser.add_argument("--load_pdf_dir", dest="LOAD_PDF_DIR", type=str, required=True, help="Path to pdf")
    args = parser.parse_args()

    tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base", do_lower_case =False)
    model = BertForTokenClassification.from_pretrained(args.LOAD_CHECKPOINT_DIR, num_labels = 25)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    with open('idx2tag.json') as json_file:
        idx2tag = json.load(json_file)

    # read data
    data = pd.read_json('Vietnamese Entity Recognition in Resumes.json', lines=True)

    example = data['content'][4]

    ner_results = nlp(example)

    tokenizers = []
    tags = []
    for word in ner_results:
        tokenizers.append(word['word'])
        tags.append(idx2tag[word['entity']])

    print(tokenizers)
    print(tags)
    result = extract_info_cons(tokenizers, tags) 
    print(result)