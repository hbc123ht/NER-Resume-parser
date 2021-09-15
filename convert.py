import pandas as pd
import json
import codecs
import re


def is_label(string, categories):

    for category in categories:
        if category in string:
            return category
    
    return None


if __name__ == '__main__':
    # read content
    data = pd.read_csv('data_label_vn.csv')
    raw_contents = data['text']

    # initiate categories
    categories = ["NAME", "BIRTH", "GENDER", "PHONE", "ADDRESS", "EMAIL"]

    cleaned_data = pd.DataFrame()

    for raw_content in raw_contents:
        raw_content = raw_content.replace(".", "") 
        raw_content = raw_content.replace("|", "") 
        raw_content = re.split('txt\n|>|<', raw_content)

        content = ""
        annotation = []
        skip = 1

        for id in range(len(raw_content)):
            if skip > 0:
                skip = skip - 1
                continue

            substring = raw_content[id]
            label = is_label(substring, categories)

            if label != None:
                start = len(content)
                content = content + raw_content[id + 1]
                end = len(content) 
                annotation.append({
                                    "label" : [label], 
                                    "points": [
                                        {
                                            "start" : start,
                                            "end" : end,
                                            "text" : raw_content[id + 1]
                                        }
                                    ]
                                })
                skip = 2
            else:
                content = content + raw_content[id]
        cleaned_data_ = {"content" : content, "annotation" : annotation}
        cleaned_data = cleaned_data.append(cleaned_data_, ignore_index=True)
        
    cleaned_data = cleaned_data.to_json('Vietnamese Entity Recognition in Resumes.json', force_ascii=False, orient='records', lines=True)
   
    

            
                    

