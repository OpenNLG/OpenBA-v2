from datasets import load_dataset
import pandas as pd
import csv
import json
import os
import re
import random
random.seed(42)
'''
PIQA 数据集介绍了物理常识推理任务和相应的基准数据集
PIQA 侧重于日常情况，偏好非典型解决方案。
该数据集的灵感来源于 instructables.com，它为用户提供了如何使用日常材料建造、制作、烘焙或摆弄物品的指导。
train/val有标签，test无标签；测试使用val dataset
'''

def load_and_save(path):
    dataset = load_dataset("piqa")
    train_df = pd.DataFrame(dataset['train'])
    train_df.to_csv(data_path+"train.csv", index=False)
    val_df = pd.DataFrame(dataset['validation'])
    val_df.to_csv(data_path+"val.csv", index=False)
    test_df = pd.DataFrame(dataset['test'])
    test_df.to_csv(data_path+"test.csv", index=False)


import pandas as pd
import csv
import json
import os
import re
import random

def csv_to_list(filepath):
    """
    Convert a CSV file to a list of questions.
    """
    questions = []
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  
        #['goal', 'sol1', 'sol2', 'label']:二选一
        # ["How do I ready a guinea pig cage for it's new occupants?", 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.', 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.', '0']
       
        for row in reader:
            # print(row)
            answer = 'B' if int(row[3]) else 'A'
            questions.append({
                'question': row[0],
                'res1': row[1] ,
                'res2': row[2] ,
                'ans': answer
            })
            # print(questions)
            # break
    return questions

def generate_json(test_path, output_path):

    test_data = csv_to_list(test_path)
    demo_list = random.sample(test_data, 10)
    final_data = []
    for item in test_data:
        entry = {
                'demo': demo_list,
                'data': item
            }
        final_data.append(entry)
        
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)


data_path='PIQA/data/'
load_and_save(data_path)

output_dir = 'PIQA/data/10shot/PIQA-Test.json'
generate_json(data_path+'val.csv', output_dir)
