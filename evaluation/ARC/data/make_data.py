import csv
import json
import os
import re
import random
random.seed(42)


def csv_to_list(filepath):
    """
    Convert a CSV file to a list of questions.
    """
    questions = []

    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  
        # ['id', 'question', 'choices', 'answerKey']
 
        for row in reader:
            opt_dict={}
            key_mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', 'A':'A', 'B':'B', 'C':'C', 'D':'D', 'E':'E'}
            
            choice = eval(row[2])
            # {'text': ['measuring the amount of fatigue', 'making sure the same foods are eaten', 'recording observations in the same chart', 'making sure the foods are at the same temperature'], 'label': ['A', 'B', 'C', 'D']}
            
            for i in range(len(choice['label'])):
                opt_dict[key_mapping[choice['label'][i]]]=choice['text'][i]
            
            question={'question': row[1],'ans': key_mapping[row[3]], 'opt_num': len(opt_dict.keys())}
            
            for i in range(len(opt_dict.keys())+1):
                if i > 0 :
                    # print('res'+str(i))
                    # print(opt_dict[chr(ord(str(i))+16)])
                    question['res'+str(i)] = opt_dict[chr(ord(str(i))+16)]
                
            questions.append(question)
    
    return questions

def generate_json(val_path, test_path, output_path):

    test_data = csv_to_list(test_path)
    demo_list = csv_to_list(val_path)
    demo_list = random.sample(demo_list, 25)
    final_data = []
    for item in test_data:
        entry = {
                'demo': demo_list,
                'data': item
            }
        final_data.append(entry)
        
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)


def generate_json(val_path, test_path, output_path):

    test_data = csv_to_list(test_path)
    demo_list = csv_to_list(val_path)
    demo_list = random.sample(demo_list, 5)
    final_data = []
    for item in test_data:
        entry = {
                'demo': demo_list,
                'data': item
            }
        final_data.append(entry)
        
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    
    arcC_path='ARC/data/ai2_arc/ARC-Challenge/'
    arcE_path='ARC/data/ai2_arc/ARC-Easy/'

    val_path = arcC_path+'val.csv'
    test_path = arcC_path+'test.csv'
    output_dir = 'ARC/data/25shot/ARC-{}-Test.json'.format("Challenge")
    generate_json(val_path, test_path, output_dir)

    val_path = arcE_path+'val.csv'
    test_path = arcE_path+'test.csv'
    output_dir = 'ARC/data/25shot/ARC-{}-Test.json'.format("Easy")
    generate_json(val_path, test_path, output_dir)

    
   

