
import sys


if len(sys.argv) != 3:
    print("Usage: python StancEval_Test.py qarib qarib/bert-base-qarib")
    sys.exit(1)

# Retrieve arguments
model_n = sys.argv[1] 
model_name = sys.argv[2]

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
    data_training_df = pd.read_csv("Data/Mawqif_All_Train.csv")
    data_val_df = pd.read_csv("Data/Mawqif_All_Val.csv")
    return {
        "train" : data_training_df.fillna("None"),
        "val" : data_val_df.fillna("None"),
    }
        

data_all_dict = get_data()

print(f"Size of trining Data: {len(data_all_dict['train'])}")
print(f"Size of validation Data: {len(data_all_dict['val'])}")
data_dict = data_all_dict.copy()

feature = 'stance'
set(data_dict['train'][feature])

mapping = {'None': 0, 'Favor': 1, 'Against': 2}
class_names = ['None','Favor','Against']
if feature == 'sentiment':
    mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    class_names = ['Negative', 'Neutral', 'Positive']
elif feature == 'sarcasm':
    mapping = {'No': 0, 'Yes': 1}
    class_names = ['No','Yes']


data_dict['train'][feature] = data_dict['train'][feature].apply(lambda x: mapping[x])
data_dict['val'][feature] = data_dict['val'][feature].apply(lambda x: mapping[x])
#test_df['stance'] = test_df['stance'].apply(lambda x: mapping[x])

LABEL_COLUMNS=[feature]


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


id2label = {v: k for k, v in mapping.items()}


test_data = pd.read_csv("./Data/Mawqif_AllTargets_Blind Test.csv")
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
    
df.to_csv(f"report_testDataset_{model_n}.csv", index=False)


with open("SMASH_STANCEEVAL_2024.csv", "w", encoding="utf-8") as fout:
    for line in output_:
        fout.write(f"{line}\n")