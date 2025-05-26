# generate dataset for train 
# dataset_name such as : fiqa, msmarco, nq  
python dense_retrieval/retriever/dpr/train/gen_data_for_colbert.py --dataset_name dataset_name --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70
python dense_retrieval/retriever/dpr/train/gen_data_for_colbert.py --dataset_name fiqa --exp_name no_aug

# train the colbert with data and pseudo data
bash dense_retrieval/retriever/col_bert/train_colbert.sh -g 0 -d fiqa -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -m 500 -s 500 -b 64
#  test
bash dense_retrieval/retriever/col_bert/test_colbert.sh -g 0 -d fiqa -e llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70 -p 60 -c 500
