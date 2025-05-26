from mono_reranker.mono import MonoT5, MonoT5Dataset,mono_rerank
import argparse






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus_file', type=str, required=True)
    parser.add_argument('-query_file', type=str, required=True)
    parser.add_argument('-ranking_file', type=str, required=True)
    parser.add_argument('-reranker_path', type=str, default="castorini/monot5-base-msmarco", help='Path to the reranker model, default is castorini/monot5-base-msmarco')

    args = parser.parse_args()
    corpus_file = args.corpus_file
    query_file = args.query_file
    ranking_file = args.ranking_file
    reranker_path = args.reranker_path
    mono_reranker = MonoT5(reranker_path)
    
    mono_rerank(corpus_file,query_file, ranking_file,mono_reranker)
    
    
    
