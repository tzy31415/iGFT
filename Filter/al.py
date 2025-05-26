import argparse
from active_learning.lossnet import train_main, test_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pseudo', type=str, required=True, help='Pseudo query file to be filtered')
    parser.add_argument('-corpus', type=str, required=True, help='Corpus File')
    parser.add_argument("-train",type=str, default="True",help="train lossnet or not")

    parser.add_argument('-tofile', type=str, help='The JSON file to write')
    parser.add_argument('-base_retriever_path', type=str, default='/root/autodl-tmp/sptar/bert/bert-base-uncased', help='The base retriever model')
    parser.add_argument('-lossnet_path', type=str, default="lossnet", help='The lossnet model name')

    args = parser.parse_args()
    pseudo_query_file = args.pseudo
    corpus = args.corpus
    to_json_file = args.tofile
    base_retriever_path = args.base_retriever_path
    lossnet_path = args.lossnet_path
    train = args.train
    
    
    
    if train == "True":
        train_main(pseudo_query_file, corpus,lossnet_path , base_retriever_path)
    else:
        test_main(pseudo_query_file, corpus, to_json_file, lossnet_path, base_retriever_path)
    
    
