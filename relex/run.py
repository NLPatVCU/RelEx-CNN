# Author : Samantha Mahendran for RelEx-CNN

import configparser
import CV, train_test
import ast
from utils import re_number, extract_entites

config = configparser.ConfigParser()
config.read('configs/n2c2.ini')

if config.getboolean('SEGMENTATION', 'no_relation'):
    no_rel_label = ast.literal_eval(config.get("SEGMENTATION", "no_rel_label"))
else:
    no_rel_label = None

test = config.getboolean('DEFAULT', 'test')
binary = config.getboolean('DEFAULT', 'binary_classification')
write_predictions = config.getboolean('DEFAULT', 'write_predictions')
write_no_relations = config.getboolean('PREDICTIONS', 'write_no_relations')
dominant_entity = ast.literal_eval(config.get("SEGMENTATION", "dominant_entity"))
rel_labels = ast.literal_eval(config.get("SEGMENTATION", "rel_labels"))
downsample_allow = config.getboolean('SEGMENTATION', 'downsample_allow')

if test:
    # binary classification
    if binary:
        print("Please note if it is binary classification predictions must be written to files")
        # write entities to the output files
        extract_entites.write_entities( config['SEGMENTATION']['test_path'], config['PREDICTIONS']['final_predictions'])
        # for each label
        for label in rel_labels[1:]:
            rel_labels = [rel_labels[0], label]
            # perform segmentation
            seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'], config['SEGMENTATION']['test_path'], rel_labels, no_rel_label, config.getboolean('SEGMENTATION', 'no_rel_multiple'), dominant_entity[0],
                                   config.getint('SEGMENTATION', 'no_of_cores'), downsample_allow,
                                         config.getint('SEGMENTATION', 'downsample_ratio'))
            train_test.run_CNN_model(seg_train, seg_test, config['CNN_MODELS']['embedding_path'], config.getint('CNN_MODELS', 'embedding_dim'),
                         config['CNN_MODELS']['model'], dominant_entity[0], write_predictions, write_no_relations,
                         config['PREDICTIONS']['initial_predictions'], config['PREDICTIONS']['binary_predictions'])

        re_number.append(config['PREDICTIONS']['binary_predictions'], config['PREDICTIONS']['final_predictions'])
    else:
        seg_train, seg_test = train_test.segment(config['SEGMENTATION']['train_path'], config['SEGMENTATION']['test_path'], rel_labels, no_rel_label,  config.getboolean('SEGMENTATION', 'no_rel_multiple'), dominant_entity[0],
                                   config.getint('SEGMENTATION', 'no_of_cores'), downsample_allow,
                                         config.getint('SEGMENTATION', 'downsample_ratio'), config['PREDICTIONS']['final_predictions'])

        train_test.run_CNN_model(seg_train, seg_test, config['CNN_MODELS']['embedding_path'], config.getint('CNN_MODELS', 'embedding_dim'),
                             config['CNN_MODELS']['model'], dominant_entity[0], write_predictions, write_no_relations,
                             config['PREDICTIONS']['initial_predictions'], config['PREDICTIONS']['final_predictions'])
else:
    seg_train = CV.segment(config['SEGMENTATION']['train_path'], rel_labels, no_rel_label,  config.getboolean('SEGMENTATION', 'no_rel_multiple'), dominant_entity[0],
                           config.getint('SEGMENTATION', 'no_of_cores'),config['PREDICTIONS']['final_predictions'])

    CV.run_CNN_model(seg_train, config['CNN_MODELS']['embedding_path'], config.getint('CNN_MODELS', 'embedding_dim'),
                     config['CNN_MODELS']['model'], dominant_entity[0], write_predictions, write_no_relations, config['PREDICTIONS']['initial_predictions'], config['PREDICTIONS']['final_predictions'])