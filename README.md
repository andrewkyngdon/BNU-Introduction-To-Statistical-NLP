# BNU Introduction To Deep Learning Workshop

The distributed representation of a word (or "word embedding") was an important contribution to statistical Natural Language Processing (NLP). This is a repository of Deep Learning language models trained as part of an introductory workshop in Deep Learning for social science students at Beijing Normal University, Beijing, China.

All models were trained with [Caffe for Windows](https://github.com/BVLC/caffe/tree/windows) using an NVIDIA GTX 1060 GPU, CUDA 8.0 and cuDNN 6.0. The official Caffe repo for any supported OS can be used with these models without modification.

## Neural Probabilistic Language Modelling

A Deep Learning model to train real-valued vector representations of words was proposed by [Bengio, Ducharme, Vincent and Jauvin (2003)](http://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). This repo contains a Caffe implementation of the original feed-forward deep neural network created by these scholars. A Netscope visualisation of the model is [here](https://ethereon.github.io/netscope/#/gist/0327dd98a7eb87de64845255a6fbd23d).

The data used to learn the language model were prepared from Assignment Two of [Geoffrey Hinton's "Neural Networks for Machine Learning" Coursera course](https://www.coursera.org/learn/neural-networks).

### Usage

Clone this repo to the desired location, then navigate to the "bengio_lm" directory. To train the model, type the following at the command prompt:

```
caffe train -solver solver_bengio.prototxt
```
Training and test error can be logged by adding either `2>&1 | tee train.log` (Linux) or `>> train.log 2>&1` (Windows) after the above command. These files can be parsed for error curve visualisation using the script `parse_log.py` in the `CAFFE_ROOT/caffe/tools/extra` directory.

The trained model can be used to predict words and to calculate the similarity of words that are in the training corpus. To do this, type `jupyter notebook` at the command prompt, then open `Bengio_Language_Model.ipynb` in your browser. Edit as necessary.

The notebook `Bengio_Language_Model.ipynb` also contains code for visualising Caffe word embeddings using the dimensionality reduction technique of [t-SNE](https://lvdmaaten.github.io/tsne/) and the `matplotlib.pyplot` Python library.

**Notes**
- The Bengio language model may require some time to train. Pre-trained weights are supplied in `BengioLM.caffemodel` if this is an issue.
- If you completed Hinton's course and kept the materials, place the [Octave](https://www.gnu.org/software/octave/download.html) file `save_for_python.m` and `ConvertHDF5.py` into the relevant directory. Running the former in Octave then executing `python ConvertHDF5.py` at the command prompt will create the csv files and HDF5 databases used here.
- Should you do the above, you can also substitute the text passage Hinton used with one of your own choosing.

## Word2Vec and pre-training word embeddings for text classification models

The Word2Vec language models, as proposed by Tomas Mikolov, et al, (2013) at Google in [this paper](https://arxiv.org/abs/1301.3781) and [this one](https://arxiv.org/abs/1310.4546) , can be used to pre-train word embeddings for use in other neural language models.

Caffe implementations of the Continuous Bag of Words (CBoW), Continuous Skip-Gram and Continuous Skip-Gram with [Negative Sampling](https://arxiv.org/abs/1402.3722) methods were used to pre-train word embeddings for a simple Automated Essay Scoring language model. The dataset used was Essay 6 from the Kaggle [Automated Student Assessment Prize](https://www.kaggle.com/c/asap-aes) of 2012.

### Usage

Navigate to the "classify_text" directory after cloning this repo. The notebook `Create Toy CBoW_Skip-Gram data.ipynb` contains a toy text passage and code to create CBoW and Skip-Gram data.

To train either the CBoW or Skip-Gram Word2Vec models, first open `Wrd2VecCaffe.prototxt` and change the data source file to either `trn_CBoW_hdf5_list.txt` or `trn_skpgrm3_hdf5_list.txt`. Then type the following at the command prompt:

```
caffe train -solver w2v_solver.prototxt
```
The Continuous Skip-Gram with Negative Sampling model uses a somewhat different architecture, a Netscope visualisation of which is [here](https://ethereon.github.io/netscope/#/gist/18cf971dc13cc75ed47cb5114bd62130). To train this model, edit the `net:` field in `w2v_solver.prototxt` to state `SkpGrmNeg.prototxt`. The HDF5 database can be downloaded from [here](https://drive.google.com/open?id=1Yhzgf3QEB0qqGdS5fd1QM3roN2gYdbDs).

Three pre-trained Automated Essay Scoring (AES) language models, based on the simple Bengio, et al, (2003) architecture, can be downloaded - [CBoW Embedding](https://drive.google.com/open?id=1hYpt0_Co7Nm5Nj7Q7pt2fxiq4VEiKzvd), [Skip-Gram Embedding](https://drive.google.com/open?id=1Q_6a0lEVuqCH5ngcrhlilXeEoNOrXPup) and [Skip-Gram-Negative-Sampling Embedding](https://drive.google.com/open?id=1mPZuLsXZKypOiis8k2qAMhHViPhzP46T). Use the notebook `aes.ipynb` to explore these; and to visualise the hidden layer activations using t-SNE.

**Notes**
- The essays used were subject to text pre-processing and cleaning, including but not limited to the removal of stop words, punctuation, numeric characters and capitalisation.
- The Kaggle "essays" are in fact short written answers, not extended response essays. These are well known to not be as conducive to AES as proper essays. 
- Human awarded marks approximate a (usually skewed) Gaussian, so class imbalance is almost always present. A dataset to train an AES for production must have an equal (and adequate) number of essays across all score points. This is not true of the Kaggle data.
- Other factors can greatly affect the application of AES, such as the skill and experience of the human examiners, the marking QA process, the acceptable mark discrepancy, whether rubric based or wholistic marking with exemplar scripts was used, and whether the essays were written online or were transcribed (the latter process can introduce much error and can thus require much cleaning of text).
