from tqdm import tqdm
import json
import random
import csv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset

class MonoT5():
    def __init__(self, model_name = 'castorini/monot5-base-msmarco') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def score(self, query, doc):
        input_text = f"Query: {query} Document: {doc}"

        input_text = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(input_text, return_dict_in_generate=True, output_scores=True,output_logits=True)
        decoded_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        prob_list = outputs.logits[0].cpu().numpy().flatten().tolist()
        true_value = prob_list[1176]
        false_value = prob_list[6136]

        if decoded_output == 'false':
            return  0
        else:
            return float(np.exp(true_value)/(np.exp(false_value)+np.exp(true_value)))







class MonoT5Dataset(Dataset):
    def __init__(self, train_data, tokenizer, max_length=512): 
        self.train_data = train_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        doc = self.train_data[idx]['corpus']
        query = self.train_data[idx]['pseudo_query']
        input_text = f"Query: {query} Document: {doc}"
        label = 1
        # input_text, label = self.train_data[idx]
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            label,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels = labels.input_ids.squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        


def read_jsonl(filename):
    id2doc = {}
    with open(filename, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            id1 = item['_id']
            text = item['text']
            id2doc[id1] = text
    return id2doc




def read_tsv(filename):
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        qid2cid = []
        for row in tqdm(reader):
            if row[2] == 'score':
                continue
            qid2cid.append([row[0],row[1]])
    return qid2cid



def write_tsv_line_by_line(data, file):
    with open(file, 'w',encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        # writer.writerow(['query-id', 'corpus-id', 'score'])
        for qid,cid,rank in data:
            writer.writerow([qid, cid, rank])



def process_qrel(data):
    qid2cid_rank = {}

    for item in data:
        qid, cid = item
        if qid not in qid2cid_rank.keys():
            qid2cid_rank[qid] = [cid]
        else:
            qid2cid_rank[qid].append(cid)
    return qid2cid_rank


def lineid2cid(corpus):
    id2doc = {}
    with open(corpus, 'r') as f:
        for idx, line in tqdm(enumerate(f)):
            item = json.loads(line)
            id1 = item['_id']
            text = item['text']
            id2doc[str(idx)] = id1
    return id2doc








def mono_rerank(corpus_file,query_file, ranking_file,mono):
    corpus_data = read_jsonl(corpus_file)
    query_data = read_jsonl(query_file)
    rank_data = read_tsv(ranking_file)
    lid2cid = lineid2cid(corpus_file)

    result = []
    rank_data = process_qrel(rank_data)
    for qid in tqdm(rank_data.keys()):
        cid_list = rank_data[qid]
        query = query_data[qid]
        temp = {}
        for cid in cid_list:
            cid1 = lid2cid[cid]
            corpus = corpus_data[cid1]
            score = mono.score(query,corpus)
            temp[cid] = score
        temp = dict(sorted(temp.items(), key=lambda item: item[1], reverse=True))
        rank = 1
        for cid in temp:
            result.append([qid,cid,rank])
            rank += 1
    write_tsv_line_by_line(result, ranking_file)





