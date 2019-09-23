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
   

## To reproduce our Results:
   * Get UFSAC datasets
   * Run commands ...





