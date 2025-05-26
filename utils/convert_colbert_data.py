import json
import csv
from tqdm import tqdm
import os
import argparse
def write_json_line_by_line(data, file):
    with open(file, 'w') as json_file:
        for element in data:
            json.dump(element, json_file)
            json_file.write('\n') 
            
def write_tsv_line_by_line(data, file):
    with open(file, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['query-id', 'corpus-id', 'score'])
        for qid,cid in data:
            writer.writerow([qid, cid, '1'])
    


def get_data_all_files(filename):
    result = []
    with open(filename, 'r') as f:
        result = json.load(f)
    return result


def read_tsv(filename):
    dq_dict = []
    with open(filename, newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        header = next(reader)  
        for row in tqdm(reader):
            dq_dict.append([row[0],row[1]])
    return dq_dict



def get_coprpus_all_files(filename):
    result = []
    with open(filename, 'r') as f:
        result = json.load(f)
    return result


def sort_query(data, sort_criterion='score'):
    data = sorted(data, key=lambda x: x[sort_criterion], reverse=True)
    return data

def get_corpus_line_by_line(filename):
    result = {}
    with open(filename, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            id1 = data['_id']
            text = data['text']
            result[text] = id1
    return result

def get_colbert_pseudo_data(pseudo_data, dataset_name):
    init_count = 4000001
    pseudo_data = get_data_all_files(pseudo_data)
    if 'score'  in pseudo_data[0].keys():
        pseudo_data = sort_query(pseudo_data) # Sort the pseudo queries by score
    if not os.path.exists(f'ColBERT/pseudo_query/data/{dataset_name}_50/100k'):
        os.makedirs(f'ColBERT/pseudo_query/data/{dataset_name}_50/100k')
    
    tsv_file = f'ColBERT/pseudo_query/data/{dataset_name}_50/100k/weak_train_50_llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70.tsv'
    jsonl_file = f'ColBERT/pseudo_query/data/{dataset_name}_50/100k/weak_queries_50_llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70.jsonl'
    tsv_data = []
    jsonl_data = []
    
    
    for i, value in tqdm(enumerate(pseudo_data)):
        cid = value['cid']
        qid = init_count 
        init_count += 1
        pseudo_query = value['pseudo_query']
        metadata = {}
        tsv_data.append([qid, cid])
        jsonl_data.append({
            "_id":qid,
            "text":pseudo_query,
            "metadata":metadata
        }) 
    write_json_line_by_line(jsonl_data, jsonl_file)
    write_tsv_line_by_line(tsv_data, tsv_file)
               
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('-pseudo', type=str, required=True, help='File name to save the pseudo data')
    args = parser.parse_args()
    pseudo = args.pseudo
    dataset = args.dataset
    get_colbert_pseudo_data(pseudo, dataset)
    
    
    
    
    