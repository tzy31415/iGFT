import argparse
import json
import csv
from tqdm import tqdm
import random


def get_corpus_line_by_line(filename): 
    result = {}
    with open(filename, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            id1 = data['_id']
            text = data['text']
            result[id1] = text
    return result

def read_tsv(filename):
    dq_dict = []
    with open(filename, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        header = next(reader) 
        for row in tqdm(reader):
            dq_dict.append([row[0],row[1]])
    return dq_dict

def write_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Successfully save to {json_file}!")
        

if __name__ == '__main__':
    # First, download the dataset using the download_dataset.sh script!!
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-num', type=int, required=True, help='Number of samples from the dataset')
    args = parser.parse_args()
    dataset_name = args.dataset
    sample_num = args.num
    
    corpus_file = f'{dataset_name}/corpus.jsonl'
    query_file = f'{dataset_name}/queries.jsonl'
    train_dict = f'{dataset_name}/qrels/train.tsv'
    corpus_dict = get_corpus_line_by_line(corpus_file)
    query_dict = get_corpus_line_by_line(query_file)
    dq_dict = read_tsv(train_dict)

    count = 0
    output_result = []
    
    if sample_num == -1:
        sample_num = len(dq_dict)
        print(f'Sample all {sample_num} data from {dataset_name} dataset.')
    
    
    for qd in tqdm(dq_dict):
        if count >= sample_num:
            break
        count += 1
        q_id, c_id = qd
        instruction = ''
        query = query_dict[q_id]
        corpus = corpus_dict[c_id]
        
        
        output_result.append({
            'corpus_id': c_id,
            'instruction': corpus,
            'input':'',
            'output': query
        })
    write_json(output_result, f'sample_{dataset_name}.json')


