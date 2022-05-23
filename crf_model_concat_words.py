# import pandas as pd
import sklearn
import re

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

import pandas as pd
import csv
from flashtext import KeywordProcessor
import pickle
from sklearn.externals import joblib


nlp = spacy.load('en')

DATA_INP = '/Users/ashwins/repo/DataScience/dailymed_crf/data/'
DATA_OP = '/Users/ashwins/repo/DataScience/dailymed_crf/inference/'

# import our class
import data_pipeline_new_multiprocessing_copy as data_pipeline
from data_pipeline_new_multiprocessing_copy import load_sents_multiprocessing

# import data_pipeline_new_indiv_word_multiprocessing_copy as data_pipeline
# from data_pipeline_new_indiv_word_multiprocessing_copy import load_sents_multiprocessing

class CRF_Model():
    
    def __init__(self, streamer, ignoreSpaces=0, feature_select = 0, c1 = 0.5, c2 = 1.2, iterations=20, algorithm = 'lbfgs'):
     
        crf = sklearn_crfsuite.CRF(
        algorithm=algorithm,
        c1=c1,
        c2=c2,
        max_iterations=iterations,
        all_possible_transitions=True,
        all_possible_states=True,
        verbose = True
        #     error_sensitive= True
        )
        
        self.feature_select = feature_select
        self.streamer= streamer
        self.model = crf
        self.ignoreSpaces = ignoreSpaces

        self.mapped_dd = pd.DataFrame([], columns = ['drug', 'disease'])
        self.df_list = []


    def load_training_data(self,  file_path = DATA_INP):
        print("Started  :: Streaming data pipeline started...")
        start = time.time()
        self.sents, self.y_sents, self.total_y_sents, self.true_x_sents, self.true_y_sents, self.false_x_sents, self.false_y_sents = load_sents_multiprocessing()
        print("time for tagging sents : ", time.time() - start)
        start = time.time()

        if self.ignoreSpaces:
            print(":: Streaming data without spaces...")
            _ = self.streamer.gen_pattern_no_spaces(keywords = None)
            self.sents, self.sents_no_spaces, self.y_sents, self.sent_indexes, self.total_y_sents = self.streamer.tag_data_no_spaces()

            self.X_train, self.y_train, self.X_test, self.y_test = self.streamer.create_train_data(self.sents_no_spaces, self.y_sents, self.sent_indexes, self.total_y_sents, self.feature_select)

        else:
            print(":: Streaming data with spaces...")
            start = time.time()
        
            self.X_train, self.y_train, self.X_test, self.y_test, self.test_sents = self.streamer.create_train_data(sents=self.sents, y_sents=self.y_sents, total_y_sents=self.total_y_sents, false_x_sents=self.false_x_sents, true_x_sents=self.true_x_sents, false_y_sents=self.false_y_sents, true_y_sents=self.true_y_sents, feature_select = self.feature_select)
        
        print("time for creating features for training data...", time.time() - start)

        return self.X_train, self.y_train, self.X_test, self.y_test


    def train(self):
        self.model.fit(self.X_train, self.y_train)
        self.labels = list(self.model.classes_)
        
        self.sorted_labels = sorted(self.labels, key=lambda name: (name[1:], name[0]))

    def get_labels(self, sort = False) : 
        self.labels = list(self.model.classes_)
        if sort :
            self.sorted_labels = sorted(self.labels, key=lambda name: (name[1:], name[0]))
            return self.sorted_labels
        else:
            return self.labels
    

    def print_report(self):
        print("Flat F1 score : ")
        print(metrics.flat_f1_score(self.y_test, self.y_pred, average='weighted', labels= self.labels))
        print("Classification Report : ")
        print(metrics.flat_classification_report(self.y_test, self.y_pred, labels=self.sorted_labels, digits=3))


    def infer_data(self):
        if self.ignoreSpaces :
            self.X_infer, self.y_true, self.sents = self.streamer.create_inference_data(self.sents_no_spaces, self.total_y_sents, feature_select= self.feature_select)
        else:
            print("sents type is: ",type(self.sents[0]))
            self.X_infer, self.y_true, self.sents = self.streamer.create_inference_data(self.sents, self.total_y_sents, feature_select = 0)
        start = time.time()
        self.y_infer = self.model.predict(self.X_infer)
        # print("len y infer : ", len(self.y_infer))
        print("\n\ntime for inference: ", time.time()-start, "\n\n")
        return self.X_infer, self.y_true, self.y_infer, self.sents


    def infer_metrics(self):
        self.X_infer, self.y_true, self.y_infer, _ = self.infer_data()
        # print("total y_sent len : ",len(total_y_sents))
        # print("total sent len : ",len(sents))
        print("Complete metrics of inference...")
        print(metrics.flat_classification_report(self.y_true, self.y_infer, labels=self.sorted_labels, digits=3))


    def print_inferred_data(self) :
        # _, y_true, sents = streamer.create_inference_data()
        _, _, self.y_infer, self.sents = self.infer_data()
        
        with open(DATA_OP + 'crf_output.csv', 'w') as op:
            op_writer = csv.writer(op)
            i=0
            for x, y in zip(self.sents, self.y_infer):
                print("X -->", x, "Y --> ", y)
                op_writer.writerow(zip(x,y))
                # op_writer.writerow(y)
                if i>=2:
                    break
                i+=1


    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        print("len y pred : , x test, y test")
        print(len(self.y_pred))
        print(len(self.X_test))
        print(len(self.y_test))
        return self.y_pred

    def print_test_results(self):
        print("Test results are : \n")
        with open(DATA_OP + 'crf_test_output_concat_word.csv', 'w') as op:            
            op_writer = csv.writer(op)
            for sent, pred in zip(self.test_sents, self.y_pred):
                op_writer.writerow(zip(sent,pred))                
                for ele in zip(sent, pred):
                    print(ele)

    def get_diseases(self, X_infer, sents):
        diseases = []
        disease_sents = self.model.predict(X_infer)
        for j, s in enumerate(disease_sents):
            for i, text in enumerate(s):
                # print(i, '\t', text)
                if text == 'yes':
                    diseases.append(sents[j][i])        
        return diseases

    def drug_disease_mapper(self):
        data_df = pd.read_csv(DATA_INP + 'input_crf_new.csv', encoding='utf-8', header=None)
        self.df_search = pd.read_csv('/Users/ashwins/Scripts/dd_map_scrape/done_data/consolidated_final.csv', encoding='utf-8', index_col=0)
        data_df.apply(lambda x : self._get_maps(x), axis=1)

        self.mapped_dd = pd.concat(self.df_list, ignore_index=True)    
        self.mapped_dd.drop_duplicates(inplace=True)    
        return self.mapped_dd

    def _get_keywords(self, drug_name):
        
        self.false_keywords = self.df_search[(self.df_search['drug'] == drug_name) & (self.df_search['type'] == 'n')]['indication'].astype('str').tolist()
        self.true_keywords = self.df_search[(self.df_search['drug'] == drug_name) & (self.df_search['type'] == 'y')]['indication'].astype('str').tolist()
        # print(type(self.true_keywords), type(self.false_keywords))

    def _get_maps(self, row):
        self._get_keywords(row[1])
        haystack = str(row[3])

        processor = KeywordProcessor()
        processor.add_keywords_from_list(keyword_list = self.true_keywords)
        processor.add_keywords_from_list(keyword_list= self.false_keywords)
        found_list = list(set(processor.extract_keywords(haystack.lower(), span_info=True)))
        found_list= sorted(found_list, key=lambda x : x[1])
        prev_len =0
        para = ''
        for i, t in enumerate(found_list):
            para += haystack[prev_len:t[1]]
            para += "_".join(haystack[t[1] : t[2]].split())
            para += ' '
            prev_len = t[2]+1
            # print("prev_len : ", prev_len)
            # print("para : ", para)
            # print("truw",t)
        # print("\nfinal para: ", para)
        para = re.sub(r"\-", "_", para)
        para = re.sub(r"\(|\)", "", para)
        sents = [s for  s in nlp(re.sub(r"\n|u\'", " ", para)).sents]
        
        X_infer = [self.streamer.sent2features(s, feature_select = 0) for s in sents]
        diseases = self.get_diseases(X_infer, sents)
        temp = pd.DataFrame(diseases)
        temp['drugs'] = str(row[1]) # drugs
        self.df_list.append(temp)

    def save_model(self):
        pickle.dump(self.model,open( DATA_OP + "trained_crf.pkl", "wb"))

    def load_model(self, model_path = DATA_OP + "trained_crf.pkl"):
        self.loaded_model = pickle.load(open(model_path, 'rb'))


def main():
    streamer = data_pipeline.Data_Stream()
    
    init_start =time.time()

    Model = CRF_Model(streamer, ignoreSpaces = 0, feature_select = 0, c1 = 0.5, c2 = 1.5, iterations=80, algorithm = 'lbfgs')
    X_train , y_train, X_test, y_test = Model.load_training_data()

    print("\nLoaded data...")
    # print(X_train[:10], y_train[:10], X_test[:10], y_test[:10])
    # crf = Model.create_model()
    print("Created model...\n\n")
    start = time.time()
    Model.train()
    y_pred = Model.predict()
    labels = Model.get_labels()
    sorted_labels = Model.get_labels( sort=True)
    
    
    Model.predict()
    Model.print_test_results()
    Model.print_report()
    
    print("time taken for training: ", time.time() - init_start)
    print("time taken for creating training data and training: ", time.time() - init_start)
    start = time.time()
    df = Model.drug_disease_mapper()
    print(df.columns)
    df.drop_duplicates(subset = ['drugs', 0], inplace=True)
    print(df)
    df.to_csv(DATA_OP + 'mapped.csv', encoding='utf-8')
    
    Model.infer_metrics()
    Model.print_inferred_data()
    print("time for complete inference is : ", time.time() - start)
    Model.save_model()
    Model.load_model()


if __name__ == '__main__' :
    main()
    