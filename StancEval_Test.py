import sys


if len(sys.argv) != 6:
    print("Usage: python StancEval_Test.py qarib qarib/bert-base-qarib trainingDataSource testDataSource OutputFile")
    sys.exit(1)

model_n = sys.argv[1] 
model_name = sys.argv[2]
training_data = sys.argv[3]
test_data = sys.argv[4]
outputFile = sys.argv[5]

import pandas as pd
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from transformers import AutoModelForSequenceClassification
import evaluate
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")
from sklearn.model_selection import train_test_split


# Load data
import sys
def get_data():
    data_training_df = pd.read_csv(training_data)
    
    return {
        "train" : data_training_df.fillna("None"),
    }
        

data_dict = get_data()

print(f"Size of trining Data: {len(data_dict['train'])}")

feature = 'stance'
set(data_dict['train'][feature])

mapping = {'None': 0, 'Favor': 1, 'Against': 2}
class_names = ['None','Favor','Against']


data_dict['train'][feature] = data_dict['train'][feature].apply(lambda x: mapping[x])

LABEL_COLUMNS=[feature]


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


id2label = {v: k for k, v in mapping.items()}


test_data = pd.read_csv(test_data)
test_data = test_data.fillna("")
test_data

df = pd.DataFrame()



output_ = ["ID\tTarget\tTweet\tStance"]
for target in test_data.target.unique():
    
    print(f"model_name: {model_name}")
    
    print(f"Target: {target}\n")
    model_path = f"./baseline_trainings/{model_n}_{target}_{feature}/model.path"  # Path where the model was saved
    
    print(f"model_path: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print(model.config.id2label)

    df = pd.DataFrame()
    data_temp = test_data[test_data['target']==target]
    for qid, text in zip(data_temp['ID'], data_temp['text']):
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        
        predicted_class_id = logits.argmax().item()

        
    
        predicted_main_class = model.config.id2label[predicted_class_id]
        
        df = pd.concat([df, pd.DataFrame.from_records([{"ID": qid, "Target": target , "Tweet": text, "Stance": predicted_main_class.upper()}])])
        output_.append(f"{qid}\t{target}\t{text}\t{predicted_main_class}")
    
df.to_csv(f"report_testDataset_{outputFile}.csv", index=False)


with open(f"{outputFile}.csv", "w", encoding="utf-8") as fout:
    for line in output_:
        fout.write(f"{line}\n")