This is the code repository for reproducing the result of our ACL 2022 paper [Modeling Syntactic-Semantic Dependency Correlations in Semantic Role Labeling Using Mixture Models](). 
We developed the code on the codebase of [Linguistically-Informed Self-Attention](https://github.com/ChristoMartin/LISA) (LISA) and added token-based batching component from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor). 


Requirements
-------------------------
- Tensorflow 1.15
- Python 3.6
- h5py (for ELMo and BERT models)

Quick start
-------------------------
## Data setup
### Obtaining data
- You need to obtain CoNLL-2009 dataset. Look at this [site](https://ufal.mff.cuni.cz/conll2009-st/index.html) for reference.
### Embeddings
- [bilm-tf](https://github.com/allenai/bilm-tf) for ELMo embeddings
- [BERT](https://huggingface.co/docs/transformers/model_doc/bert) for BERT embeddings
- [FastText](https://fasttext.cc/) 
- [GloVe](https://nlp.stanford.edu/projects/glove/)
--------
We have packed the pipeline into several scripts in conll09-all_langs
To prepare the data, you need to do the following in the directory of a specific language:
- put the train&dev&test file into the respective directory
- execute the followings
```
make rename_as_conll05 section=$(train/dev/test)
make all_parse section=$(train/dev/test)
make gather_all_info 
make correct_synt_idx
```

Running experiments
-------------------------
Please download the saved checkpoint [here], extracting the checkpoits
run the command stored in the "eval.cmd" file. For example, you can find the following command in `conll-eng-mm5-fasttext.zip`.
```
bin/evaluate-exported.sh config/llisa/e2e/fasttext/conll09-eng-sa-small-dep_prior-par_inp-bilinear-gp-ll.conf --save_dir <path_to_model>/best_checkpoint --num_gpus 1 --hparams  mixture_model=5
```
