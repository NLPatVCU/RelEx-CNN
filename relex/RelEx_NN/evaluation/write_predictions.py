# Author : Samantha Mahendran for RelEx-CNN

import os
import numpy as np
import pandas as pd


class Predictions:
    def __init__(self, initial_predictions, final_predictions, No_Rel=False, dominant_entity='F'):
        """
        Write predictions back to files
        :param final_predictions: predicted relations
        :param No_Rel: flag whether to write the relations with No-relation label back to files
        """

        self.dominant_entity = dominant_entity
        self.No_Rel = No_Rel
        print("No rel:", self.No_Rel)
        # path to the file which tracks the entity pair details
        input_track = 'track.npy'
        # path to the file which tracks the predicted la
        input_pred = 'pred.npy'
        # path to the folder to save the predictions
        self.initial_predictions = initial_predictions
        # path to the folder to save the re-ordered predictions to the files where the entities are already appended
        self.final_predictions = final_predictions

        # Delete all files in the folder before the prediction
        ext = ".ann"
        filelist = [f for f in os.listdir(self.initial_predictions) if f.endswith(ext)]
        for f in filelist:
            os.remove(os.path.join(self.initial_predictions, f))

        # load the numpy files
        self.track = np.load(input_track)
        self.pred = np.load(input_pred)

        self.write_relations()
        self.renumber_relations()

    def write_relations(self):
        """
        write the predicted relations into their respective files
        :param track: tracking information of the relation from the original
        :param pred: relation predictions
        """
        for x in range(0, self.track.shape[0]):
            # check the length of file name and padding
            if len(str(self.track[x, 0])) == 1:
                file = "000" + str(self.track[x, 0]) + ".ann"
            elif len(str(self.track[x, 0])) == 2:
                file = "00" + str(self.track[x, 0]) + ".ann"
            elif len(str(self.track[x, 0])) == 3:
                file = "0" + str(self.track[x, 0]) + ".ann"
            else:
                file = str(self.track[x, 0]) + ".ann"
            # key for relations (not for a document but for all files)
            key = "R" + str(x + 1)
            # entity pair in the relations
            e1 = "T" + str(self.track[x, 1])
            e2 = "T" + str(self.track[x, 2])
            f1 = open(self.initial_predictions + str(file), "a")
            if self.No_Rel:
                # predicted label for the relation
                label = self.pred[x]
                # open and append relation the respective files in BRAT format
                f1.write(str(key) + '\t' + str(label) + ' ' + 'Arg1:' + str(e1) + ' ' + 'Arg2:' + str(e2) + '\n')
                f1.close()
            else:
                # predicted label for the relation
                label = self.pred[x]
                if label != 'No-Relation':
                    # open and append relation the respective files in BRAT format
                    if self.dominant_entity == 'F':
                        f1.write(str(key) + '\t' + str(label) + ' ' + 'Arg1:' + str(e1) + ' ' + 'Arg2:' + str(e2) + '\n')
                    else:
                        f1.write(str(key) + '\t' + str(label) + ' ' + 'Arg1:' + str(e2) + ' ' + 'Arg2:' + str(e1) + '\n')
                    f1.close()

    def renumber_relations(self):
        """
        When writing predictions to file the key of the relations are not ordered based on individual files.
        This function renumbers the appended predicted relations in each file
        """
        for filename in os.listdir(self.initial_predictions):
            print(filename)
            if os.stat(self.initial_predictions + filename).st_size == 0:
                continue
            else:
                df = pd.read_csv(self.initial_predictions + filename, header=None, sep="\t")
                df.columns = ['key', 'body']
                df['key'] = df.index + 1
                df['key'] = 'R' + df['key'].astype(str)
                df.to_csv(self.final_predictions + filename, sep='\t', index=False, header=False, mode='a')
