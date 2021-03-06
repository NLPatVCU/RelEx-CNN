# Author : Samantha Mahendran for RelEx-CNN

from keras.layers import *
from keras.models import *
from sklearn.model_selection import StratifiedKFold
from RelEx_NN.evaluation import evaluate
from RelEx_NN.evaluation import Predictions
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class Segment_CNN:

    def __init__(self, model, embedding, cross_validation=False, end_to_end=False, epochs=20, batch_size=512,
                 filters=32, filter_conv=1, filter_maxPool=5, activation='relu', output_activation='sigmoid', drop_out=0.5,
                 loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'], dominant_entity='S', initial_predictions=None, final_predictions=None,
                 write_No_rel=False):

        """
        Builds and run Segment CNN model
        :param model: data after prepocessing
        :param embedding: word embeddings
        :param cross_validation: flag to perform CV (default fold = 5)
        :param initial_predictions: folder to save the initial relation predictisons
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
            self.data_model.write_Predictions = True

    def define_model(self):
        """
        define a CNN model with defined parameters when the class is called
        :param no_classes: no of classes
        :return: trained model
        """
        input_shape = Input(shape=(self.data_model.maxlen,))
        embedding = Embedding(self.data_model.common_words, self.embedding.embedding_dim,
                              weights=[self.embedding.embedding_matrix], trainable=False)(input_shape)
        conv = Conv1D(filters=self.filters, kernel_size=self.filter_conv, activation=self.activation)(embedding)
        drop = Dropout(self.drop_out)(conv)
        pool = MaxPooling1D(pool_size=self.filter_maxPool)(drop)
        flat = Flatten()(pool)

        return flat, input_shape

    def build_segment_cnn(self, no_classes):
        """
        Builds individual units for each segments
        :param no_classes: no of classes
        :return: trained model
        """
        flat1, input_shape1 = self.define_model()
        flat2, input_shape2 = self.define_model()
        flat3, input_shape3 = self.define_model()
        flat4, input_shape4 = self.define_model()
        flat5, input_shape5 = self.define_model()

        # merge
        merged = concatenate([flat1, flat2, flat3, flat4, flat5])

        # interpretation
        dense1 = Dense(18, activation=self.activation)(merged)
        outputs = Dense(no_classes, activation=self.output_activation)(dense1)

        model = Model(inputs=[input_shape1, input_shape2, input_shape3, input_shape4, input_shape5], outputs=outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # summarize
        print(model.summary())
        return model

    def test(self):
        # train - data segments
        pre_train = self.data_model.preceding
        mid_train = self.data_model.middle
        suc_train = self.data_model.succeeding
        c1_train = self.data_model.concept1
        c2_train = self.data_model.concept2
        y_train = self.data_model.train_label

        binary_y_train = self.data_model.binarize_labels(y_train, True)

        # test data segments
        pre_test = self.data_model.test_preceding
        mid_test = self.data_model.test_middle
        suc_test = self.data_model.test_succeeding
        c1_test = self.data_model.test_concept1
        c2_test = self.data_model.test_concept2
        track_test = self.data_model.test_track

        if not self.data_model.write_Predictions:
            y_test = self.data_model.y_test
            binary_y_test = self.data_model.binarize_labels(y_test, True)

        labels = [str(i) for i in self.data_model.encoder.classes_]

        cv_model = self.build_segment_cnn(len(self.data_model.encoder.classes_))
        cv_model.fit([pre_train, mid_train, suc_train, c1_train, c2_train], binary_y_train, epochs=self.epochs, batch_size=self.batch_size)

        if self.data_model.write_Predictions:
            pred = evaluate.predict_test_only(cv_model, [pre_test, mid_test, suc_test, c1_test, c2_test], labels)
            # save files in numpy to write predictions in BRAT format
            np.save('track', np.array(track_test))
            np.save('pred', np.array(pred))
            Predictions(self.initial_predictions, self.final_predictions, self.write_No_rel)
            # Predictions(self.initial_predictions, self.final_predictions, self.write_No_rel, dominant_entity=self.dominant_entity)
        else:
            y_pred, y_true = evaluate.predict(cv_model, [pre_test, mid_test, suc_test, c1_test, c2_test], binary_y_test,
                                              labels)
            print(classification_report(y_true, y_pred, labels=labels))
            print(confusion_matrix(y_true, y_pred))

    def cross_validate(self, num_folds=5):
        """
        Perform cross validation
        :param num_folds: no of fold for cross validation (default = 5)
        """
        Pre_data = self.data_model.preceding
        Mid_data = self.data_model.middle
        Suc_data = self.data_model.succeeding
        C1_data = self.data_model.concept1
        C2_data = self.data_model.concept2
        Track = self.data_model.train_track
        Y_data = self.data_model.train_label

        if num_folds <= 1: raise ValueError("Number of folds for cross validation must be greater than 1")

        skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
        skf.get_n_splits(C1_data, Y_data)
        evaluation_statistics = {}
        fold = 1
        originalclass = []
        predictedclass = []
        # to track the entity pairs for each relation
        brat_track = []

        for train_index, test_index in skf.split(C1_data, Y_data):
            binary_Y = self.data_model.binarize_labels(Y_data, True)

            pre_train, pre_test = Pre_data[train_index], Pre_data[test_index]
            mid_train, mid_test = Mid_data[train_index], Mid_data[test_index]
            suc_train, suc_test = Suc_data[train_index], Suc_data[test_index]
            c1_train, c1_test = C1_data[train_index], C1_data[test_index]
            c2_train, c2_test = C2_data[train_index], C2_data[test_index]
            track_train, track_test = Track[train_index], Track[test_index]
            y_train, y_test = binary_Y[train_index], binary_Y[test_index]

            labels = [str(i) for i in self.data_model.encoder.classes_]

            cv_model = self.build_segment_cnn(len(self.data_model.encoder.classes_))
            cv_model.fit([pre_train, mid_train, suc_train, c1_train, c2_train], y_train, epochs=self.epochs,
                         batch_size=self.batch_size)
            y_pred, y_true = evaluate.predict(cv_model, [pre_test, mid_test, suc_test, c1_test, c2_test], y_test,
                                              labels)
            fold_statistics = evaluate.cv_evaluation_fold(y_pred, y_true, labels)
            originalclass.extend(y_true)
            predictedclass.extend(y_pred)
            brat_track.extend(track_test)

            print("--------------------------- Results ------------------------------------")
            print(classification_report(y_true, y_pred, labels=labels))
            evaluation_statistics[fold] = fold_statistics
            fold += 1

        if self.data_model.write_Predictions:
            # save files in numpy to write predictions in BRAT format
            np.save('track', np.array(brat_track))
            np.save('pred', np.array(predictedclass))
            Predictions(self.initial_predictions, self.final_predictions, self.write_No_rel, dominant_entity=self.dominant_entity)

        print("--------------------- Results --------------------------------")
        print(classification_report(np.array(originalclass), np.array(predictedclass), target_names=labels))
        print(confusion_matrix(np.array(originalclass), np.array(predictedclass)))

