# Word Sense Disambiguation using BERT, ELMo and Flair

### Overview:
This repository contains the implementation of our approach proposed in the paper

> Wiedemann, G., Remus, S., Chawla, A., Biemann, C. (2019): [Does BERT Make Any Sense? Interpretable Word Sense Disambiguation with Contextualized Embeddings. Proceedings of KONVENS 2019](https://www.inf.uni-hamburg.de/en/inst/ab/lt/publications/2019-wiedemannetal-bert-sense.pdf), Erlangen, Germany.


Word Sense Disambiguation (WSD) is the task to identify the correct sense of the usage of a word from a (usually) fixed inventory of sense identifiers.
We propose a simple approach that scans through the training data to learn the Contextualized Word Embeddings(CWE) of sense labels and classifies the ambiguous words on the basis Cosine Similarity with the learnt CWEs for that word.

## Requirements:
   * [Python3.5 or above](https://www.python.org/)
   * [PyTorch 0.4.0](https://pytorch.org/)
   * [Huggingface's Pytorch-Pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
   * [Allennlp ELMo](http://www.allennlp.org/)
   * [Flair](https://github.com/zalandoresearch/flair)
   * [Scikit-learn](https://scikit-learn.org/)
   * [UFSAC](https://github.com/getalp/UFSAC)
   * [Java (version 8 or higher)](https://java.com)
   * [Maven](https://maven.apache.org)
   

## Dataset:

This section will guide you through the steps to reproduce all the results that were mentioned in the paper.

We use the following corpus files from the UFSAC repository in our experiments:


| Dataset Name | Training File Name  | Testing File Name |
|--|--|--|
| Senseval 2 | senseval2_lexical_sample_train.xml  | senseval2_lexical_sample_test.xml |
|Senseval 3|senseval3task6_train.xml|senseval3task6_test.xml|
| Semcor | semcor.xml  | - |
| WNGT | wngt.xml  | - |
| Semeval 2007 task 7 | - | semeval2007task7.xml |
| Semeval 2007 task 17  | -| semeval2007task17.xml |

Before we proceed to see the steps to reproduce our results, please note a few points:
* In our experiments, we use Semcor and WNGT separately for training Wordnet sense identifiers and test them individually across Semeval 2007 task 7 and 17.
* Testing on Senseval 2 and Senseval 3 does not involve Semcor and WNGT for training. We only use their respective training datasets to train the Wordnet sense identifiers.

## To reproduce our results:

The repository is composed of three python files:

 - [BERT_Model.py](https://github.com/uhh-lt/bert-sense/blob/master/BERT_Model.py "BERT_Model.py")
 - [ELMO_Model.py](https://github.com/uhh-lt/bert-sense/blob/master/ELMO_Model.py "ELMO_Model.py")
 - [Flair_Model.py](https://github.com/uhh-lt/bert-sense/blob/master/Flair_Model.py "Flair_Model.py")

The steps to carry out the experiments are same for all the files. We shall demonstrate how to reproduce the results for BERT model and you can follow the same steps for the other 2 files as well.

Run `python BERT_Model.py` with the following main arguments:

 - `--use_cuda [True|False]`: (default True) provide whether you want to switch the training to GPU or not.
 - `--device [cuda:0 | cuda:1 ...]` : (default 'cuda:2') Specify the device number on GPU you want to load the BERT model to and do the training.
 - `--train_corpus [FILE_NAME]`: This is the path to your training file.
 - `--train_type [SE|SEM|WNGT]`: Specify this argument depending upon the dataset you are using for training. (The values are case-sensitive).

| Dataset | Argument value |
|--|--|
|Semeval  | SE |
|Semcor  | SEM |
|WNGT  | WNGT |
 - `--trained_pickle [FILE_NAME]`: Save/Load the dictionary of generated BERT embeddings in a pickle file to avoid recomputing them again and again.
 - `--test_corpus [FILE_NAME]`: This is the path to your testing file. Make sure that you provide the correct testing files corresponding to your training file. *Look into Table 1 of this README file to understand better.*
 - `--start_k [N]` : (default 1) The lower value of the parameter k(the number of nearest neighbors).
 - `--end_k [N]` : (default 1) The upper value of the parameter k(the number of nearest neighbors). Code shall be executed for every k in the range [start_k, end_k].
 - `--save_xml_to [FILE_NAME]`: This is the file where you wish to store your final predictions on test data. (for every iteration starting from *k=start_k* to *k=end_k*, output would be FILE_NAME_k.
 - `--use_euclidean [1|0]` : (default 0) Use Euclidean Distance instead of Cosine Similarity to find the nearest neighbors. 
 -  `--reduced_search [1|0]` : (default 0) This uses POS information of the word attribute in the XML files to find the nearest neighbor of a particular word only among the sense identifier that posses its POS tag. Setting this to 1 gives rise to the POS-sensitive kNN model. 
 
**You may follow the same steps as mentioned above to reproduce the results for Flair and ELMo Model using `Flair_Model.py` and `ELMO_Model.py` files respectively.**

<br />

Below, we provide two examples to understand how to use the arguments mentioned above:

 1. To generate the results for SE-2 as mentioned in Table 3 of the paper, run the `BERT_Model.py` file as follows: 
 
 `python BERT_Model.py --use_cuda=True --device=cuda:0 --train_corpus=senseval2_lexical_sample_train.xml trained_pickle=BERT_embs.pickle --test_corpus=senseval2_lexical_sample_test.xml --start_k=1 --end_k=10 --save_xml_to=SE2.xml --use_euclidean=0 --reduced_search=0`

2. To generate the results for S7-T7 on WNGT as mentioned in Table 5 of the paper, run the `BERT_Model.py` file as follows: 

 `python BERT_Model.py --use_cuda=True --device=cuda:0 --train_corpus=wngt.xml trained_pickle=WNGT_BERT_embs.pickle --test_corpus=semeval2007task7.xml --start_k=1 --end_k=10 --save_xml_to=SE7T17.xml --use_euclidean=0 --reduced_search=1`

## Evaluation

The final output is stored in an XML file where the '**word**' tag will now have an extra attribute named '**WSD**'.  This will be our model's prediction. To test the accuracy, we use a script in the UFSAC repository. Following are the steps to obtain Precision, Recall, F1 score, etc. for our prediction file.

 -  Find the `evaluate_wsd.sh` script in the UFSAC repository.
 - Run `evaluate_wsd.sh --corpus=[PATH-TO-PREDICTION-FILE] --reference_tag=wn30_key --hypothesis_tag=WSD`. 
 
