This is the code repository for reproducing the result of our ACL 2022 paper [Modeling Syntactic-Semantic Dependency Correlations in Semantic Role Labeling Using Mixture Models](). 
We developed the code on the codebase of [Linguistically-Informed Self-Attention](https://github.com/ChristoMartin/LISA) (LISA) and added token-based batching component from [Tensor2Tensor](). 


Requirements
-------------------------
- Tensorflow 1.15
- Python 3.6
- h5py (for ELMo and BERT models)
You can also use our singularity container avaiable [here]() for easy environment setup.

Quick start
-------------------------
Data setup
--------
- You need to obtain CoNLL-2009 dataset first and run 

Embeddings
--------
- We use [bilm-tf](https://github.com/allenai/bilm-tf) for extracting ELMo embeddings
- We use [BERT](https://huggingface.co/docs/transformers/model_doc/bert) for extracting BERT embeddings
- You can find FastText [here](https://fasttext.cc/) 
- You can find GloVe [here](https://nlp.stanford.edu/projects/glove/)
--------
We have packed the pipeline into several scripts in conll09-all_langs
To prepare the data, you need to do the following in the directory of a specific language:
- put the train&dev&test file into the respective directory
execute the followings
- make rename_as_conll05 section=$(train/dev/test)
- make all_parse section=$(train/dev/test)
- make gather_all_info 
- make correct_synt_idx
  

Running experiments
-------------------------
Please download the saved checkpoint [here], extracting the checkpoits