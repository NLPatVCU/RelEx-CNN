# Author : Samantha Mahendran for RelEx-CNN

from RelEx_NN.model import Model
from RelEx_NN.embeddings import Embeddings
from RelEx_NN.cnn import Segment_CNN
from RelEx_NN.cnn import Sentence_CNN
from segment import Set_Connection


def segment(train, test, entites, no_rel=None, dominant_entity='S', no_rel_multiple=False, parallelize=False,
            no_of_cores=64, predictions_folder=None):
    """
    Perform segmentation for the training and testing data
    :param dominant_entity:
    :param write_Entites:
    :param parallelize:
    :param no_of_cores:
    :param train: path to train data
    :param test: path to test data
    :param entites: list of entities that create the relations
    :param no_rel: name the label when entities that do not have relations in a sentence are considered
    :param no_rel_multiple: flag whether multiple labels are possibles for No-relation
    :param predictions_folder: path to predictions (output) folder
    :return: segments of train and test data
    """

    if no_rel:
        seg_train = Set_Connection(CSV=False, dataset=train, rel_labels=entites, no_labels=no_rel,
                                   no_rel_multiple=no_rel_multiple, dominant_entity=dominant_entity,
                                   parallelize=parallelize, no_of_cores=no_of_cores,
                                   predictions_folder=predictions_folder).data_object
    else:
        print("Start segmentation of train set")
        seg_train = Set_Connection(CSV=False, dataset=train, rel_labels=entites, no_rel_multiple=no_rel_multiple, dominant_entity=dominant_entity, parallelize=parallelize, no_of_cores=no_of_cores,
                                   predictions_folder=predictions_folder).data_object
    print("Start segmentation of test set")
    seg_test = Set_Connection(CSV=False, dataset=test, rel_labels=entites, test=True, no_rel_multiple=no_rel_multiple, dominant_entity=dominant_entity, parallelize=parallelize, no_of_cores=no_of_cores, predictions_folder=predictions_folder).data_object

    return seg_train, seg_test


def run_CNN_model(seg_train, seg_test, embedding_path, embed_dim, cnn_model, dominant_entity, write_predictions=False,
                  write_No_rel=False, initial_predictions=None, final_predictions=None):
    """
    Call CNN models
    :param write_predictions: write entities and predictions to file
    :param dominant_entity:
    :param embedding_path:
    :param seg_test: test segments
    :param cnn_model: choose the model
    :param seg_train: train segments
    :param initial_predictions: folder to save the initial relation predictions
    :param final_predictions: folder to save the final relation predictions
    :param write_No_rel: Write the no-relation predictions back to files
    :return: None
    """
    if cnn_model == 'segment':
        model = Model(data_object=seg_train, data_object_test=seg_test, segment=True, test=True,
                      write_Predictions=write_predictions)
        embedding = Embeddings(embedding_path, model, embedding_dim=embed_dim)
        seg_cnn = Segment_CNN(model, embedding, dominant_entity = dominant_entity, initial_predictions=initial_predictions,
                              final_predictions=final_predictions, write_No_rel=write_No_rel)

    else:
        model = Model(data_object=seg_train, data_object_test=seg_test, test=True, write_Predictions=write_predictions)
        embedding = Embeddings(embedding_path, model, embedding_dim=embed_dim)
        single_sent_cnn = Sentence_CNN(model, embedding, dominant_entity = dominant_entity, initial_predictions=initial_predictions,
                                       final_predictions=final_predictions, write_No_rel=write_No_rel)
