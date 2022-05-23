# import pandas as pd
import sklearn

from itertools import chain
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import spacy

import numpy as np
import time

import crf_model
import data_pipeline
# from fuzzywuzzy import fuzz

nlp = spacy.load('en')

DATA_INP = '/Users/ashwins/Scripts/notebooks/synonym_crf/'

class MultiModel():

    def __init__(self):
        self.streamer = data_pipeline.Data_Stream()
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def create_feature_list(self):
        self.feature_select_list = []

    def multi_CRFs(self, num_crf) :
        self.CRFs= []

        for i in range(num_crf):
            try:
                self.CRFs.append(crf_model.CRF_Model(self.streamer, self.feature_select_list[i]))
            except :
                self.CRFs.append(crf_model.CRF_Model(self.streamer))

            print("Created %.2f CRFs...\n\n" % num_crf)

    def single_CRF(self, feature_select = 0):
        self.CRF = crf_model.CRF_Model(self.streamer, feature_select = 0)
        print("Created CRF...\n\n")

    def single_load_training_data(self, model=None):
        if not model:
            model = self.CRF

        X_train , y_train, X_test, y_test = model.load_training_data(self.streamer)
        print("Loaded data...\n\n")
        # print(X_train[:10], y_train[:10], X_test[:10], y_test[:10])
        return X_train, y_train, X_test, y_test

    def single_train(self, model = None):
        if not model :
            model = self.CRF

        model.train()

    def single_pred(self, model = None):
        if not model :
            model = self.CRF

        y_pred = model.predict()
        return y_pred

    def single_print_metrics(self, model = None):
        if not model:
            model = self.CRF

        model.print_report()
        model.infer_metrics()

    def single_inference(self, model = None):
        if not model:
            model = self.CRF

        model.print_inferred_data()

    def multi_load_training_data(self):
        for i in range(len(self.CRFs)):
            tmp_xtrain, tmp_ytrain, tmp_xtest, tmp_ytest = self.single_load_training_data(self.CRFs[i])
            self.X_train.append(tmp_xtrain)
            self.y_train.append(tmp_ytrain)
            self.X_test.append(tmp_xtest)
            self.y_test.append(tmp_ytest)
        # pass

    def multi_train(self) :

        pass

    def multi_pred(self) :
        pass

    def multi_print_metrics(self) :
        pass

def main():
    mm = MultiModel()
    mm.single_CRF(feature_select=0)
    mm.single_load_training_data()
    mm.single_train()
    mm.single_pred()
    mm.single_print_metrics()

if __name__ == '__main__':
    main()