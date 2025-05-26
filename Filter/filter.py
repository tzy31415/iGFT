from sparse_filter.sparse_filter import get_bm_score
from dense_filter.dense_filter import get_dense_score
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pseudo', type=str, required=True, help='Pseudo query file to be filtered')
    parser.add_argument('-corpus', type=str, required=True, help='The dataset corpus, specifically in BEIR format')
    parser.add_argument('-tofile', type=str, required=True, help='The JSON file to write')
    parser.add_argument('-mode', type=str, required=True, help='BM25, DPR, ColBERT, MonoT5')
    parser.add_argument('-candidate_num', type=int, default=0)

    args = parser.parse_args()
    pseudo_query_file = args.pseudo
    corpus_file = args.corpus
    to_json_file = args.tofile
    mode = args.mode
    candidate_num = args.candidate_num
    if mode == 'BM25':
        get_bm_score(pseudo_query_file, corpus_file, to_json_file, candidate_num)
    elif mode in ['DPR', 'ColBERT', 'MonoT5']:
        get_dense_score(pseudo_query_file, corpus_file, to_json_file, mode, candidate_num)
    else:
        assert False, 'Invalid mode'
    
    
    
    
    