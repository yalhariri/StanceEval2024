import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python StancEval_Test.py testDataSource OutputFile")
    sys.exit(1)

test_data = sys.argv[1]
outputFile = sys.argv[2]

test_data = pd.read_csv(test_data)
test_data = test_data.fillna("None")
test_data

test_set_dict = {item['ID']: item for item in test_data.to_dict(orient="index").values()}

mapper={'Women empowerment':'تمكين المرأة',
'Covid Vaccine':'لقاح كوفيد',
'Digital Transformation':'التحول الرقمي'}


import ollama
from tqdm import tqdm_notebook as tqdm


model_name = "command-r:35b-v0.1-q8_0"
system_context = "You are an expert in analysing people's opinions. You are an expert in Arabic. \
# You will be given an Arabic sentence as input. Your task is to identify the stance towards the topic or subject discussed in the sentence. \
# Your task is to identify whether the sentence is in favour of the topic, against it, or neutral. \
Your output should be one of the following: favour, against, or neutral. You should not provide any further information. Your answer should be in English."
question = "What is the stance in the given sentence? [favour, against, neutral]."

counter = 0
progress_bar = tqdm(test_set_dict.keys(), desc="Processing")
for k in progress_bar:
    text = test_set_dict[k]['text']
    topic = test_set_dict[k]['target']
    msg=f"الجملة: {text}\nالموضوع:{mapper[topic]}\n{question}"
    
    message=[{
        'role': 'system',
        'content':system_context},
        {
            'role':'user',
            'content': msg
        }]
    response = ollama.chat(model=model_name, messages=message)
    #print(response['message']['content'])
    #print('-'*20)
    test_set_dict[k]['label'] = response['message']['content']
    progress_bar.update(1)
progress_bar.close()



output_ = ["ID\tTarget\tTweet\tStance"]
for k in test_set_dict.keys():
    if 'favour' in test_set_dict[k]['label'].lower():
        label = "favor".upper()
    elif 'against' in test_set_dict[k]['label'].lower():
        label = "against".upper()
    else:
        label = "None".upper()
    output_.append(f"{test_set_dict[k]['ID']}\t{test_set_dict[k]['target']}\t{test_set_dict[k]['text']}\t{label}")


with open(f"{outputFile}.csv", "w", encoding="utf-8") as fout:
    for line in output_:
        fout.write(f"{line}\n")