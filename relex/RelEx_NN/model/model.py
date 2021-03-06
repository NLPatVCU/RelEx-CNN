# Author : Samantha Mahendran for RelEx

from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def create_validation_data(train_data, train_label, num_data=1000):
    """
    Splits the input data into training and validation. By default it takes first 1000 as the validation.
    :param num_data: number of files split as validation data
    :param train_label: list of the labels of the training data
    :param train_data: list of the training data
    :return:train samples, validation samples
    """

    x_val = train_data[:num_data]
    x_train = train_data[num_data:]

    y_val = train_label[:num_data]
    y_train = train_label[num_data:]

    return x_train, x_val, y_train, y_val


class Model:

    def __init__(self, data_object, data_object_test=None, segment=False, test=False,
                 write_Predictions=False, generalize=False, de_sample=False, common_words=10000,
                 maxlen=100):
        """
        :param data_object: training data object
        :param data_object_test: testing data object (None -during 5 CV)
        :param segment: Flag to be set to activate segment-CNN (default-False)
        :param test: Flag to be set to validate the model on the test dataset (default-False)
        :type write_Predictions: write entities and predictions to file
        :param common_words: Number of words to consider as features (default = 10000)
        :param maxlen: maximum length of the vector (default = 100)

        """
        self.de_sample = de_sample
        self.write_Predictions = write_Predictions
        self.generalize = generalize
        self.segment = segment
        self.test = test
        self.common_words = common_words
        self.maxlen = maxlen
        self.data_object = data_object
        self.data_object_test = data_object_test

        # read dataset from external files
        train_data = data_object['sentence']
        train_labels = data_object['label']
        # tracks the entity pair details for a relation
        train_track = data_object['track']

        # test files only
        if self.test:
            test_data = data_object_test['sentence']
            test_labels = data_object_test['label']
            # tracks the entity pair details for a relation
            test_track = data_object_test['track']

            # to read in segments
            if segment:
                test_preceding = data_object_test['seg_preceding']
                test_middle = data_object_test['seg_middle']
                test_succeeding = data_object_test['seg_succeeding']
                test_concept1 = data_object_test['seg_concept1']
                test_concept2 = data_object_test['seg_concept2']
                self.test_preceding, self.test_middle, self.test_succeeding, self.test_concept1, self.test_concept2, self.word_index = self.vectorize_segments(
                    test_data, test_preceding, test_middle, test_succeeding, test_concept1, test_concept2)
        else:
            # when running only with train data
            test_data = None
            test_labels = None

        self.train_label = train_labels
        self.train_track = np.asarray(train_track).reshape((-1, 3))

        if self.test:
            self.train, self.x_test, self.word_index = self.vectorize_words(train_data, test_data)
            self.y_test = test_labels
            self.test_track = np.asarray(test_track).reshape((-1, 3))
        else:
            self.train, self.word_index = self.vectorize_words(train_data, test_data)

        # divides train data into partial train and validation data
        self.x_train, self.x_val, self.y_train, self.y_val = create_validation_data(self.train, self.train_label)

        if self.segment:
            train_preceding = data_object['seg_preceding']
            train_middle = data_object['seg_middle']
            train_succeeding = data_object['seg_succeeding']
            train_concept1 = data_object['seg_concept1']
            train_concept2 = data_object['seg_concept2']

            # convert into segments
            self.preceding, self.middle, self.succeeding, self.concept1, self.concept2, self.word_index = self.vectorize_segments(
                train_data, train_preceding, train_middle, train_succeeding, train_concept1, train_concept2)

    def vectorize_words(self, train_list, test_list=None):

        """
        Takes training data as input (test data is optional), creates a Keras tokenizer configured to only take into account the top given number
        of the most common words in the training data and builds the word index. If test data is passed it will be tokenized using the same
        tokenizer and output the vector. If the one-hot flag is set to true, one-hot vector is returned if not vectorized sequence is returned
        :param train_list: train data
        :param test_list: test data
        :return: one-hot encoding or the vectorized sequence of the input list, unique word index
        """
        tokenizer = Tokenizer(self.common_words)

        # This builds the word index
        tokenizer.fit_on_texts(train_list)

        # Turns strings into lists of integer indices.
        train_sequences = tokenizer.texts_to_sequences(train_list)
        padded_train = pad_sequences(train_sequences, maxlen=self.maxlen)

        if self.test:
            test_sequences = tokenizer.texts_to_sequences(test_list)
            padded_test = pad_sequences(test_sequences, maxlen=self.maxlen)

        # To recover the word index that was computed
        word_index = tokenizer.word_index

        if self.test:
            return padded_train, padded_test, word_index
        else:
            return padded_train, word_index

    def vectorize_segments(self, sentences, preceding, middle, succeeding, concept1, concept2):
        """
        Takes in the sentences and segments and creates Keras tokenizer to return the vectorized segments
        :param sentences: sentences
        :param preceding: preceding segment
        :param middle: middle
        :param succeeding: succeeding
        :param concept1: concept1
        :param concept2: concept2
        :return: vectorized segments
        """
        tokenizer = Tokenizer(self.common_words)
        # This builds the word index
        tokenizer.fit_on_texts(sentences)

        preceding_sequences = tokenizer.texts_to_sequences(preceding)
        padded_preceding = pad_sequences(preceding_sequences, maxlen=self.maxlen)

        middle_sequences = tokenizer.texts_to_sequences(middle)
        padded_middle = pad_sequences(middle_sequences, maxlen=self.maxlen)

        succeeding_sequences = tokenizer.texts_to_sequences(succeeding)
        padded_succeeding = pad_sequences(succeeding_sequences, maxlen=self.maxlen)

        concept1_sequences = tokenizer.texts_to_sequences(concept1)
        padded_concept1 = pad_sequences(concept1_sequences, maxlen=self.maxlen)

        concept2_sequences = tokenizer.texts_to_sequences(concept2)
        padded_concept2 = pad_sequences(concept2_sequences, maxlen=self.maxlen)

        # To recover the word index that was computed
        word_index = tokenizer.word_index

        return padded_preceding, padded_middle, padded_succeeding, padded_concept1, padded_concept2, word_index

    def binarize_labels(self, label_list, binarize=False):
        """
        Takes the input list and binarizes or vectorizes the labels
        If the binarize flag is set to true, it binarizes the input list in a one-vs-all fashion and outputs
        the one-hot encoding of the input list
        :param binarize: binarize flag
        :param label_list: list of text labels
        :return list:list of binarized / vectorized labels
        """

        if self.test or binarize:
            self.encoder = preprocessing.MultiLabelBinarizer()
            encoder_label = self.encoder.fit_transform([[label] for label in label_list])
        else:
            self.encoder = preprocessing.LabelEncoder()
            encoder_label = self.encoder.fit_transform(label_list)
        return encoder_label
