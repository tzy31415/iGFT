# iGFT For Low Resource Setting

This is the implementation of the paper 'From Missteps to Mastery: Enhancing Low-Resource Dense Retrieval through Adaptive Query Generation'

## Download Dataset

Download the [BEIR](https://github.com/beir-cellar/beir) dataset using 'utils/download_dataset.sh'

```bash
wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip
```

`dataset_name` can be nq, fiqa and others.

## Initial Data Sample

To simulate a low-resource scenario, we randomly sampled few examples from the training set of the dataset as the initial training data for the large language model. Additionally, we converted these samples into a format suitable for supervised fine-tuning of the large language model. The following code demonstrates this process:

```python
python utils/sample_init_data.py -dataset dataset_name -num sample_num
```

In this process, `dataset_name` refers to the dataset being sampled, which includes datasets from BEIR, such as MSMARCO, FiQA, NQ, and others. Ensure that `dataset_name` points to a valid dataset directory that has already been downloaded using the    `utils/download_dataset.sh`  script.

## LLM-based Query Generation

### Setup Llama Factory Environment

To initialize the development environment for `llama_factory`, follow the steps below:

```bash
cd llama_factory
pip install -e .
```

### **Dataset Format Alignment**

To ensure compatibility with the `llama_factory` pipeline, place the preprocessed dataset files into the `data/`directory. Subsequently, update the `data/dataset_info.json `file by appending metadata entries corresponding to the newly added dataset files. `dataset_name `is an identifier used in your training or evaluation scripts, and `file_name `is the filename of the dataset located in the `data/` directory

```json
 "dataset_name": {
    "file_name": "file_name.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
```

### Supervised Fine-Tuning

We use the LLaMA-Factory framework to optimize our query generator. First, we perform supervised fine-tuning using the initial data mentioned above. To conduct the supervised fine-tuning, use the following script under the `llama_factory/` directory :

```bash
llamafactory-cli train examples/iGFT/SFT/llama2_sft.yaml
```

The detailed configuration of training parameters can be customized in the `llama2_sft.yaml` file, including the LLM (`model_name_or_path`) to be used , the identifier of the training dataset (`dataset`), as well as other training-related hyperparameters and settings.

### Pseudo Query Generation

In our framework, the large language model serves as the query generator backbone to generate pseudo queries. We use the following script under the `llama_factory/` directory to generate queries with the optimized large language model:

```bash
llamafactory-cli generation examples/iGFT/Generation/llama2_generation.yaml 
```

The `llama2_generation.yaml` file contains the detailed configuration for the generation process. Within this file, the `adapter_name_or_path` parameter specifies the path to the optimized adapter used during inference.

### Reward Model Learning

Afterward, we score the quality of the pseudo queries using the following filtering modules. Based on the filtering results, we then iteratively optimize the large language model via reinforcement learning.

First, train the necessary reward model for reinforcement learning with the following code:

```bash
llamafactory-cli train examples/iGFT/RM/llama2_lora_reward.yaml
```

### PPO-based RL Fine-Tuning Phase

We use the PPO algorithm in combination with the previously trained reward model to perform reinforcement learning.

```bash
llamafactory-cli train examples/iGFT/PPO/llama2_lora_ppo.yaml
```

## Multi-Stage Data Filtering

The quality of the generated pseudo queries cannot be assured intrinsically. To mitigate this limitation, we introduce a set of filtering modules that assess the generated queries from multiple complementary perspectives, including sparse retrieval signals, dense semantic representations, and active learning-based uncertainty estimation.

### Filtering with Sparse Retrieval

As an initial step, we apply a sparse retrieval-based filtering module to evaluate the quality of the generated pseudo queries. The core idea is to mix the expected target document into a set of candidate documents, and then use the pseudo query to perform retrieval with a sparse retriever such as `BM25`. The parameter candidate_num defines the total number of documents in the candidate set used during evaluation.

```python
python Filter/filter.py -pseudo pseudo_file -corpus corpus_file -tofil to_write_file -mode BM25 -candidate_num 500
```

### Filtering with Dense Retrieval

The dense data quality filtering module introduces a dense retriever, aiming to filter data quality from different perspectives. We support multiple filter backbones, including `DPR`, `ColBERT`, and `MonoT5`, which can be selected based on the desired retrieval paradigm. The code is as follows,

```python
python Filter/filter.py -pseudo pseudo_file -corpus corpus_file -tofil to_write_file -mode DPR/ColBERT/MonoT5
```

### Filtering with Loss Prediction Module

The active learning-based module considers how much a pseudo query improves the performance of the retriever being trained. Here, we use prediction loss to train this active learning retriever, starting by obtaining the retriever through a pre-training method.

```bash
python Filter/al.py -pseudo pseudo_file -corpus corpus_file  -train True
```

Afterward, use the pre-trained loss predictor to predict the potential loss changes caused by the pseudo query.

```bash
python Filter/al.py -pseudo pseudo_file -corpus corpus_file -train False -tofile filter_file
```

## ColBERT Training & Validation

Afterward, the ColBERT  is trained using the filtered pseudo queries. We follow the SPTAR methodology to train our ColBERT.

First, organize the data and convert it into the appropriate format, where `dataset_name` represents the dataset name to be formatted.

```bash
python utils/convert_colbert_data.py -dataset dataset_name -pseudo filtered_pseudo_data
```

```python
cd ColBERT
python dense_retrieval/retriever/dpr/train/gen_data_for_colbert.py --dataset dataset_name --exp_name exp_name 
```

Next, train the ColBERT model, where `cuda_num` is the CUDA device number used,  `max_step` is the maximum number of training steps, and `save_per_step` is the frequency at which the model is saved.

```bash
bash  dense_retrieval/retriever/col_bert/train_colbert.sh -g cuda_num -d dataset_name -e exp_name -m max_step -s save_per_step -b batch_size
```

Finally, use this ColBERT model for retrieval and to validate the model's performance

```bash
bash  dense_retrieval/retriever/col_bert/test_colbert.sh -g cuda_num -d dataset_name -eexp_name -p par -c step
```

## **Post-Retrieval Reranking**

After completing the retrieval stage, we apply a reranking process to further refine the retrieved results and improve ranking accuracy. We adopt MonoT5, a  T5 base model that has been pre-trained on the MSMARCO dataset, as our reranker. The code for reranking using the reranker model is as follows, `ranking_file` is the result of dense retrieval.

```python
 python Filter/reranker.py -corpus_file corpus_file -query_file query_file -ranking_file ranking_file
```

# iGFT For Zero Shot Setting

In contrast to the low-resource scenario, in a zero-shot scenario, there is no initial data available for supervised fine-tuning of the query generator. Therefore, we use a zero-shot prompt approach to generate pseudo queries. The code is as follows, where `llama2_generation_zero_shot.yaml` is the configuration file for the generation process:

```bash
llamafactory-cli generation examples/iGFT/Generation/llama2_generation_zero_shot.yaml 
```

In the zero-shot scenario, apart from query generation, the other steps are the same as in the low-resource scenario and will not be elaborated on further.

# iGFT in Fully-Supervised Setting

In the fully-supervised setting, we initialize the training framework by sampling the entire training set from the original dataset to construct the initial training corpus.  The corresponding script is as follows:

```bash
python utils/sample_init_data.py -dataset dataset_name -num -1
```

# Acknowledgments

The code for the large language model fine-tuning and reinforcement learning in this work is based on the [llama-factory](https://github.com/hiyouga/LLaMA-Factory.git) repository, while the dense retrieval part is based on the [SPTAR](https://github.com/zhiyuanpeng/SPTAR.git) repository. Thanks for their wonderful works.
