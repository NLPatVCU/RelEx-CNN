# RelEx CNN documentation 
This is a deep learning-based approach to extract and classify clinical relations. This approach introduces 2 Convolutional Neural Network (CNN) models. 
Convolutional neural  networks  (CNNs)  have  been trending  due  to their  strong  learning  ability of features without manual feature engineering. Initially the convolution layer is a filter which is a set of learnable weights learned using the backpropagation algorithm and it extracts features from the input text. Maxpooling operations use the position information of local features relative to the entity pair and helps to extract the most significant feature from the output of the convolution filter. These advantages of the CNN can be utilized to reduce the dependency on manual feature engineering and learn the features automatically. 

Entity pairs of a relation are normally located in a sentence and we can represent the context of each relation by extracting  the sentence. But one sentence can include multiple distinct mentions of relations, therefore learning the entire sentence at once would not help in determining different relation classes. The sentence can be explicitly divided into segments based on the location and the context of the entities and these segments play different roles in determining the class.

Our system mainly consists  of  two  components:  Single label Sentence-CNN  and Segment-CNN.
In the following, the algorithm is explained in detail and a walk thorugh guide is provided to run the package.

## Table of Contents
1. [Installation](#installation)
   1. [Deployment](#deployment)
2. [Algorithm](#algorithm)
   1. [Data Segmentation](#data_segmentation)
   2. [Pre-processing](#pre-processing)
      1. [Vectorization](#Vectorization)
      2. [Label Binarization](#binarizer)
   3. [Word Embeddings](#word_embeddings)
   4. [CNN Models](#models)
      1. [Sentence CNN](#sen_cnn)
      2. [Segment Cnn](#seg_cnn)
   5. [Regularization](#Regularization)

## Installation

Create a python 3.6 virtual environment and install the packages given in the requirements.txt
```
pip install requirements.txt
```
### Deployment
Sample dataset (from n2c2-2018 corpus) and external embeddings are provided (/sample). 

Edit the configs file to set the paths and parameters 
```relex/N2C2/configs/n2c2.ini```

Run the following program: 
```
python relex/run_N2C2.py
```

## Algorithm 
### Data segmentation <a name="data_segmentation"></a>
Text and annotation files (in BRAT format) of a dataset are filtered and passed into a dataset object which is read in along with the entity mentions of each relation category. The dataset is prepossessed to convert it to the standard format before segmentation by removing punctuation, accent marks and converting all letters to lowercase.

The segmentation module, identifies and extracts sentences where entities are located. Sentences are divided into the following segments and wrapped into a segmentation object:
-   preceding segment
-   concept1 segment
-   middle segment
-   concept2 segment
-   succeeding segment

When extracting sentences, it checks whether the annotated relation type already exists, if not the sentences are labeled as a no-relation pair.

### Pre-processing
#### Vectorization 

Neural  networks  learn  information  through  numerical representation of the data. We need to convert the text into real number vectors. We cannot feed lists of integers into a neural network, therefore, we have to turn our lists into tensors. There are two ways to vectorize the words.

-Pad lists to have same length, and turn them into an integer tensor of shape (samples, word_indices). We used Keras tokenizer to take into account only the top given number of the most common words in data and builds a word index. We used multiple methods to vectorize the words. Maximum length of a sequence is determined and the output sequence is padded according to it. Sequences that are shorter than determined length are padded with value at the end whereas sequences longer are truncated so that they fit the desired length. Position of the padding is controlled by the arguments.

#### Label Binarization<a name="binarizer"></a>
Binarizes labels in a one-vs-all fashion. Several regression and binary classification algorithms are available in scikit-learn can be utilized for this. It converts multi-class labels to binary labels (belong or does not belong to the class) by assigning a unique value or number to each label in a categorical feature.

*Eg: labels - TrCP, TrIP, TeRP*

*TrCP 		1 0 0 0 0*

*TrIP			0 1 0 0 0*

*TeRP			0 0 1 0 0*

If the multilabel flag is set to true, then the binarization is done in the following manner:
TrCP, TrIP, TeRP	1 1 1 0 0

### Word embeddings<a name="word_embeddings"></a>
The word embeddings map a set of words or phrases in a vocabulary to real-valued vectors which helps to reduce the dimensionality and learn linguistic patterns in the data. Given a batch of vector sequences as input, the embedding layer converts the sequence into real-valued embedding vectors. Initially the weights are assigned randomly and gradually they are adjusted through backpropagation.

Using pre-trained word embeddings as features on CNN based methods have helped to achieve better performance in previous NLP related studies. We applied both Word2Vec and GloVe representations to train word embeddings in the experiments.

### CNN Models<a name="models"></a>
#### Sentence CNN <a name="sen_cnn"></a>

Each relation consists of a pair of entities and the Sentence CNN learns the relation representation for the entire sentence as a whole. First, we identify the sentence where each relation is located and extract the sentence and we feed it into a CNN for learning.

Following figure shows the function of a single label sentence CNN.

![](https://lh6.googleusercontent.com/VzMboSkKWKdFSI3E66RiL_s0NLlLJDEGQhbEywKXEIqOnWTHm39w1vPiqy3EUr5NdxRh4q375ejzX-K-znAEifHd-UZnG517UGX11G0y7j2sBb5TD4s-SWWJ2Ptq9GqK1nEZP33c)

#### Segment CNN <a name="seg_cnn"></a>
Based on where the entities are located in the sentence we can divide the sentence into segments. Different segments play different roles in determining the relation class. But each relation in Sentence-CNN is represented by an entire sentence and does not capture the positional information of the entity pairs, therefore when a sentence is divided into segments and trained by separate convolutional units.

A Sentence is explicitly segmented into 5 segments:
-   preceding - tokenized words before the first concept
-   concept 1 - tokenized words in the first concept
-   middle - tokenized words between the 2 concepts
-   concept 2 - tokenized words in the second concept
-   succeeding - tokenized words after the second concept

![](https://lh5.googleusercontent.com/_eS0O7NU9XaTM8NoO0-6ETLMF379pv25M0K22PLtni0mX5eskWrQuy196S4RA9gajiZ9zuUVIolVgO-y_iAl6hp-01jBM856rojESO1YwWIJA3oZfygQ3y5DwmdPoDdG04pMWoeD)
As the figure above shows, we construct separate convolution units for each segment and concatenate before the fixed length vector is fed to the dense layer that performs the classification.

We experiment with different sliding window sizes, filter sizes, word embeddings, loss functions to fine tune the above three models.

### Regularization

Sentence CNN and Segment CNN perform well with small filter sizes while Segment CNN performs large filter sizes. Both single and multi label Sentence CNN performed well with GloVe word embeddings whereas Segment CNN with MIMIC word embeddings.

For regularization of the model we use dropout technique on the output of convolution layer. Dropout randomly drops few nodes to prevent co-adaptation of hidden units and we set this value to 0.5 while training. We use Adam and rmsprop techniques to optimize our loss function.

# Citation
If you use this model in your work please cite as follows:
```
@article{mahendran2021extracting,
  title={Extracting Adverse Drug Events from Clinical Notes},
  author={Mahendran, Darshini and McInnes, Bridget T},
  journal={arXiv e-prints},
  pages={arXiv--2104},
  year={2021}
}
 We will update the BibTeX with the conference publication soon.
