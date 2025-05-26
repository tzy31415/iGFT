from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from tqdm import tqdm
import random

def TFIDFScore(query:str, document:str):
    tfidf_matrix = TfidfVectorizer().fit_transform([query, document])
    cosine_similarity = (tfidf_matrix[0] @ tfidf_matrix[1].T).toarray()[0][0]
    return cosine_similarity # The output range is [0,1], where 1 means very similar and 0 means very different


def BM25Score(query, gold_corpus, corpus):
    corpus = [gold_corpus] + corpus
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm_corpus = BM25Okapi(tokenized_corpus)
    score = list(bm_corpus.get_scores(query.split(" ")))
    return score[0] if score.index(max(score)) == 0 else 0 # score if found, 0 if not

def read_jsonl(filename):
    # Accepts a pseudo query to be filtered; the input file must be in JSON format.
    id2doc = {}
    with open(filename, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            # The input pseudo query file must be a list of dictionaries, and each dictionary must contain the keys 'text' and '_id'.
            id1 = item['_id']
            text = item['text']
            id2doc[id1] = text
    return id2doc

def get_coprpus_all_files(filename):
    # Read the corpus file in BEIR format.
    result = []
    with open(filename, 'r') as f:
        result = json.load(f)
    return result

def write_json(data, json_file):
    # Write the data to a JSON file.
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


def get_bm_score(pseudo_query_file, corpus_file, to_json_file,  bm_num = 500):
    corpus_data = read_jsonl(corpus_file)
    query_data = get_coprpus_all_files(pseudo_query_file)
    
    result = []
    selected_texts = random.sample(list({cid: text for cid, text in corpus_data.items()}.values()), bm_num)

    for query_item in tqdm(query_data):
        generated_query = query_item['pseudo_query']
        gold_id = query_item['cid']
        gold_corpus = query_item['corpus']
        
        score = BM25Score(generated_query,gold_corpus,selected_texts)
        result.append({
           'score':score,
           'pseudo_query':generated_query,
           'corpus':gold_corpus,
           'cid':gold_id 
        })
        
    write_json(result, to_json_file)
    
    
    