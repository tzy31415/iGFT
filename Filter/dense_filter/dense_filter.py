from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
from torch.nn.functional import softmax
import json
from tqdm import tqdm
import random

class DRPScore():
    def __init__(self,model_save_path = 'bert-base-uncased') -> None:
        self.model = DRES(models.SentenceBERT(model_save_path), batch_size=256, corpus_chunk_size=100000)
        self.retriever = EvaluateRetrieval(self.model, k_values=[10], score_function="cos_sim")
        
    def score(self, query, gold_corpus, corpus):
        query_id = "qid"
        query_list = {query_id:query, 'a':'b'}
        corpus = [gold_corpus] + corpus
        corpus = {str(i): {"text": doc} for i, doc in enumerate(corpus)}
        
        result = self.retriever.retrieve(corpus, query_list)
        score = list(result[query_id].values())
        if score.index(max(score)) == 0:
            return score[0]
        else:
            return 0

class ColBERT(): # bert，colbert
    def __init__(self, model_save_path = 'bert-base-uncased'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        self.model = AutoModel.from_pretrained(model_save_path).to(self.device)

    def embed(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sentence_embeddings

    def score(self, query, gold_corpus, doc):
        doc = [gold_corpus] + doc
        query_embeddings = self.embed(query)
        doc_embeddings = self.embed(doc)
        query_embeddings = query_embeddings.expand_as(doc_embeddings)
        similarities_list = F.cosine_similarity(query_embeddings, doc_embeddings, dim=1).tolist()
        return  similarities_list[0] if similarities_list.index(max(similarities_list)) == 0 else 0


class MonoT5():
    def __init__(self, model_name = 'castorini/monot5-base-msmarco') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
    def score(self, query, doc, doc1):
        input_text = f"Query: {query} Document: {doc}"
        input_text = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
    
        with torch.no_grad():
            outputs = self.model.generate(input_text, return_dict_in_generate=True, output_scores=True,output_logits=True)
        decoded_output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True) # truefalse
        
        prob_list = outputs.logits[0].cpu().numpy().flatten().tolist()
        true_value = prob_list[1176]
        false_value = prob_list[6136]
    
        if decoded_output == 'false':
            return  0
        else:
            return float(np.exp(true_value)/(np.exp(false_value)+np.exp(true_value)))
     


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


def get_dense_score(pseudo_query_file, corpus_file, to_json_file, mode = 'monoT5', bm_num = 500):
    corpus_data = read_jsonl(corpus_file)
    query_data = get_coprpus_all_files(pseudo_query_file)
    if mode == 'DPR':
        model = DRPScore()
    elif mode == 'ColBERT':
        model = ColBERT()
    elif mode == 'MonoT5':
        model = MonoT5()
    else:
        assert False, "Unknown mode"
    
    result = []
    selected_texts = random.sample(list({cid: text for cid, text in corpus_data.items()}.values()), bm_num)

    for query_item in tqdm(query_data):
        generated_query = query_item['pseudo_query']
        gold_id = query_item['cid']
        gold_corpus = query_item['corpus']
        
        
        score = model.score(generated_query,gold_corpus,selected_texts)
        result.append({
           'score':score,
           'pseudo_query':generated_query,
           'corpus':gold_corpus,
           'cid':gold_id 
        })
        
    write_json(result, to_json_file)
    
    
    