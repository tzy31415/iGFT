
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader


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

def get_train_data(pseudo_query_file, corpus_file):
    corpus_data = read_jsonl(corpus_file)
    query_data = get_coprpus_all_files(pseudo_query_file)
    result = []
    for query_item in tqdm(query_data):
        generated_query = query_item['pseudo_query']
        gold_id = query_item['cid']
        gold_corpus = query_item['corpus']
        
        negative_id = [id_ for id_ in corpus_data.keys() if id_ != gold_id]
        negative_id = random.choice(negative_id)
        negative_corpus = corpus_data[negative_id]
        result.append({
           'query': generated_query,
           'positive_corpus': gold_corpus,
           'negative_corpus': negative_corpus,
           'cid': gold_id 
        })
    return result

class LossDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, pos_doc, neg_doc = self.data[idx]['query'], self.data[idx]['positive_corpus'], self.data[idx]['negative_corpus']
        cid = self.data[idx]['cid']
        return {
            "query": query,
            "pos_doc": pos_doc,
            "neg_doc": neg_doc,
            "cid": cid
        }

    

class LossNet(nn.Module):
    def __init__(self, input_dim=768, interm_dim = 256):
        super(LossNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.FC1 = nn.Linear(input_dim, interm_dim).to(self.device)
        self.FC2 = nn.Linear(input_dim, interm_dim).to(self.device)
        self.FC3 = nn.Linear(input_dim, interm_dim).to(self.device)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(3 * interm_dim, 1).to(self.device)
    
    def forward(self,  q_embed, p_embed, n_embed):
        q_embed = self.FC1(q_embed)
        q_embed = self.relu(q_embed)
        q_embed = self.dropout(q_embed)
        p_embed = self.FC2(p_embed)
        p_embed = self.relu(p_embed)
        p_embed = self.dropout(p_embed)
        n_embed = self.FC3(n_embed)
        n_embed = self.relu(n_embed)
        n_embed = self.dropout(n_embed)
        features = torch.cat((q_embed, p_embed, n_embed), dim=1)  # Concatenate along the feature dimension
        features = self.linear(features)
        features = self.relu(features)
        return features
    
class ColBERT(nn.Module): # bert，colbert
    def __init__(self, model_save_path = 'bert-base-uncased'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        self.model = AutoModel.from_pretrained(model_save_path).to(self.device)

    def encode(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128).to(self.device)
        output = self.model(**encoded_input).last_hidden_state  # [B, L, H]
        attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(output.size()).float()
        embeddings = torch.sum(output * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        return embeddings  # [B, H]

    def forward(self, query, pos_doc, neg_doc):
        q_embed = self.encode(query)
        p_embed = self.encode(pos_doc)
        n_embed = self.encode(neg_doc)

        pos_score = F.cosine_similarity(q_embed, p_embed)
        neg_score = F.cosine_similarity(q_embed, n_embed)
        return pos_score, neg_score, q_embed, p_embed, n_embed
    
    
    
def train_main(pseudo_query_file, corpus_file,lossnet_path = 'lossnet', model_save_path = 'bert-base-uncased', batch_size=4, epochs=10):
    colbert = ColBERT(model_save_path)
    lossnet = LossNet()  # Assuming BERT base model with 768 hidden size
    optimizer = torch.optim.AdamW(lossnet.parameters(), lr=1e-5)
    mse_loss_fn = nn.MSELoss()
    
    dataset = LossDataset(get_train_data(pseudo_query_file, corpus_file))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    

    lossnet.train()
    for _ in range(epochs):
        for batch in tqdm(dataloader):
            query = batch["query"]
            pos_doc = batch["pos_doc"]
            neg_doc = batch["neg_doc"]

            pos_score, neg_score,  q_embed, p_embed, n_embed = colbert(query, pos_doc, neg_doc)
            target = torch.ones_like(pos_score).to(colbert.device)
            per_sample_loss = torch.clamp(-target * (pos_score - neg_score), min=0.0)  # shape: (batch_size,)

            loss_label = per_sample_loss.view(-1, 1)  # reshape to (batch_size, 1)
            loss_predict = lossnet(q_embed, p_embed, n_embed)
            loss_mse = mse_loss_fn(loss_predict, loss_label)
            loss_mse.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.save(lossnet.state_dict(), f'{lossnet_path}.pt')

    print(f"Model saved to {lossnet_path}.pt!")
def write_json(data, json_file):
    # Write the data to a JSON file.
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)     

def test_main(pseudo_query_file, corpus_file, to_json_file, lossnet_path = 'lossnet', model_save_path = 'bert-base-uncased', batch_size=1):
    colbert = ColBERT(model_save_path)
    lossnet = LossNet()
    lossnet.load_state_dict(torch.load(f'{lossnet_path}.pt'))
    lossnet.eval()
    
    dataset = LossDataset(get_train_data(pseudo_query_file, corpus_file))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            query = batch["query"]
            pos_doc = batch["pos_doc"]
            neg_doc = batch["neg_doc"]
            cid = batch["cid"]

            _, _, q_embed, p_embed, n_embed = colbert(query, pos_doc, neg_doc)
            loss_predict = lossnet(q_embed, p_embed, n_embed).cpu().numpy().tolist()
            for i in range(batch_size):
                result = {
                    'pseudo_query': query[i],
                    'corpus': pos_doc[i],
                    'negative_corpus': neg_doc[i],
                    'cid': cid[i],
                    'score': loss_predict[i][0]
                }
                results.append(result)
            
    write_json(results, to_json_file)
    
    print(f"Results saved to {to_json_file}!")