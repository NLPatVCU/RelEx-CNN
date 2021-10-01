# Author : Samantha Mahendran for RelEx-CNN
from keras.layers import *
from keras.models import *
from sklearn.model_selection import StratifiedKFold
from RelEx_NN.evaluation import evaluate
from RelEx_NN.model import model
from RelEx_NN.evaluation import Predictions
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class Sentence_CNN:

    def __init__(self, model, embedding, cross_validation=False, end_to_end=False, epochs=20, batch_size=512,
                 filters=32, filter_conv=1, filter_maxPool=5, activation='relu', output_activation='sigmoid',
                 drop_out=0.3, loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], dominant_entity='S', initial_predictions=None,
                 final_predictions=None, write_No_rel=False):
        """
        Builds and run Sentence CNN model
        :param model: data after prepocessing
        :param embedding: word embeddings
        :param cross_validation: flag to perform CV (default fold = 5)
        :param initial_predictions: folder to save the initial relation predictions
        :param final_predictions: folder to save the final relation predictions
        :param write_No_rel: Write the no-relation predictions back to files
        :param end_to_end: for experimental purpose
        """
        self.dominant_entity = dominant_entity
        self.data_model = model
        self.embedding = embedding
        self.cv = cross_validation
        self.write_No_rel = write_No_rel
        self.initial_predictions = initial_predictions
        self.final_predictions = final_predictions
        self.end_to_end = end_to_end

        # model parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.filters = filters
        self.filter_conv = filter_conv
        self.filter_maxPool = filter_maxPool
        self.activation = activation
        self.output_activation = output_activation
        self.drop_out = drop_out
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

        if self.cv:
            self.cross_validate()
        elif self.end_to_end:
            self.end_to_end_test()
        else:
            self.test()

    def define_model(self):

        input_shape = Input(shape=(self.data_model.maxlen,))
        if self.embedding:
            embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                                  weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)
        else:
            embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(input_shape)
        conv1 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(embedding)
        pool1 = MaxPooling1D(pool_size=self.filter_maxPool)(conv1)

        conv2 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(pool1)
        drop = Dropout(self.drop_out)(conv2)

        flat = Flatten()(drop)
        return flat, input_shape

    def model_without_Label(self, no_classes):
        """
        define a CNN model with defined parameters when the class is called
        :param no_classes: no of classes
        :return: trained model
        """
        # Define the model with different parameters and layers when running for multi label sentence CNN

        input_shape = Input(shape=(self.data_model.maxlen,))
        embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim)(input_shape)

        if self.embedding:
            embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                                  weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)
        conv1 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(embedding)
        pool1 = MaxPooling1D(pool_size=self.filter_maxPool)(conv1)

        conv2 = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(pool1)
        drop = Dropout(self.drop_out)(conv2)

        flat = Flatten()(drop)
        dense1 = Dense(self.filters, activation=self.activation)(flat)
        outputs = Dense(no_classes, activation=self.output_activation)(dense1)

        model = Model(inputs=input_shape, outputs=outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # summarize
        print(model.summary())

        return model

    def test(self):
        """
        Train - Test - Split
        """
        x_train = self.data_model.train
        y_train = self.data_model.train_label
        binary_y_train = self.data_model.binarize_labels(y_train, True)

        labels = [str(i) for i in self.data_model.encoder.classes_]

        x_test = self.data_model.x_test
        track_test = self.data_model.test_track
        if not self.data_model.write_Predictions:
            y_test = self.data_model.y_test
            binary_y_test = self.data_model.binarize_labels(y_test, True)

        cv_model = self.model_without_Label(len(self.data_model.encoder.classes_))
        cv_model.fit(x_train, binary_y_train, epochs=self.epochs, batch_size=self.batch_size)
        # cv_model = self.fit_Model(cv_model, x_train, binary_y_train)
        if self.data_model.write_Predictions:
            pred = evaluate.predict_test_only(cv_model, x_test, labels)

            # save files in numpy to write predictions in BRAT format
            np.save('track', np.array(track_test))
            np.save('pred', np.array(pred))
            Predictions(self.initial_predictions, self.final_predictions, self.write_No_rel, dominant_entity=self.dominant_entity)
        else:
            y_pred, y_true = evaluate.predict(cv_model, x_test, binary_y_test, labels)
            print(confusion_matrix(y_true, y_pred))
            print(classification_report(y_true, y_pred, target_names=labels))

    def cross_validate(self, num_folds=5):
        """
        Train the CNN model while running cross validation.
        :param num_folds: no of fold in CV (default = 5)
        """
        X_data = self.data_model.train
        Y_data = self.data_model.train_label

        if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")

        assert X_data is not None and Y_data is not None, \
            "Must have features and labels extracted for cross validation"

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        skf.get_n_splits(X_data, Y_data)

        evaluation_statistics = {}
        fold = 1

        originalclass = []
        predictedclass = []

        # to track the entity pairs for each relation
        brat_track = []

        Track = self.data_model.train_track

        for train_index, test_index in skf.split(X_data, Y_data):
            binary_Y = self.data_model.binarize_labels(Y_data, True)
            y_train, y_test = binary_Y[train_index], binary_Y[test_index]
            x_train, x_test = X_data[train_index], X_data[test_index]
            train_track, test_track = Track[train_index], Track[test_index]

            labels = [str(i) for i in self.data_model.encoder.classes_]

            cv_model = self.model_without_Label(len(self.data_model.encoder.classes_))
            cv_model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
            # cv_model = self.fit_Model(cv_model, x_train, y_train)
            y_pred, y_true = evaluate.predict(cv_model, x_test, y_test, labels)

            originalclass.extend(y_true)
            predictedclass.extend(y_pred)
            brat_track.extend(test_track)

            print("--------------------------- Results ------------------------------------")
            print(classification_report(y_true, y_pred, labels=labels))

            fold_statistics = evaluate.cv_evaluation_fold(y_pred, y_true, labels)

            evaluation_statistics[fold] = fold_statistics
            fold += 1
        print("--------------------- Results --------------------------------")
        print(classification_report(np.array(originalclass), np.array(predictedclass), target_names=labels))
        print(confusion_matrix(np.array(originalclass), np.array(predictedclass)))

        if self.data_model.write_Predictions:
            # save files in numpy to write predictions in BRAT format
            np.save('track', np.array(brat_track))
            np.save('pred', np.array(predictedclass))
            Predictions(self.initial_predictions, self.final_predictions, self.write_No_rel, dominant_entity=self.dominant_entity)

