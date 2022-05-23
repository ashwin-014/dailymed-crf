import pandas as pd
import sklearn
import csv
import json
from flashtext import KeywordProcessor
from itertools import chain
import spacy
# from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
import numpy as np
from fuzzywuzzy import fuzz
from sqlalchemy import create_engine, MetaData
import time
import re
import copy
from sklearn.model_selection import train_test_split
import pickle

nlp = spacy.load('en')

DATA_INP = '/Users/ashwins/repo/DataScience/dailymed_crf/data/'
DATA_STORE = '/Users/ashwins/repo/DataScience/dailymed_crf/train_data/'

class Data_Stream():
    
    def __init__(self, file_path = DATA_INP):
        self.data = ""        
        self.data_df = pd.read_csv(DATA_INP + 'oncology_input_data.csv', encoding='utf-8', header=None)
        pd.set_option('max_columns' , 20)
        print(self.data_df)
        self.sents = []
        self.y_sents= []
        self.sent_indexes = []
        self.total_y_sents = []
        self.total_sent_number = 0

        self.df_search = pd.read_csv('/Users/ashwins/Scripts/dd_map_scrape/done_data/consolidated_final.csv', encoding='utf-8', index_col=0)
        
        # data_df.apply(lambda x: self.sents.append([sent for sent in nlp(re.sub(r"\\n|u\'", " ", x[2])).sents]), axis=1) # re.sub(r"[^A-Za-z0-9 ]", " ", 
        self.data_df.apply(lambda row: self.sequence_handler(row), axis=1) # re.sub(r"[^A-Za-z0-9 ]", " ", 
        print(len(self.sents))

        print("Loaded doc and sents...")

    def sequence_handler(self, row):
        sents= [sent for sent in nlp(re.sub(r"\n|u\'", " ", row[3])).sents]
        sents_list = [[word.text for word in s] for s in sents]
        
        self.gen_pattern(row[1])
        y_sents, sent_indexes, total_y_sents = self.tag_data(sents)
        self.write_to_pickle(row[1], sents_list, y_sents, total_y_sents, sent_indexes)

    def get_keywords(self, drug_name):
        
        self.false_keywords = self.df_search[(self.df_search['drug'] == drug_name) & (self.df_search['type'] == 'n')]['indication'].tolist()
        self.true_keywords = self.df_search[(self.df_search['drug'] == drug_name) & (self.df_search['type'] == 'y')]['indication'].tolist()
        

    def gen_pattern(self, drug_name):
        print("generating patterns... \n")
        start = time.time()
        self.true_matcher = Matcher(nlp.vocab)
        self.false_matcher = Matcher(nlp.vocab)
    
        self.get_keywords(drug_name)

        for keyword in self.true_keywords :
            i=0
            pattern_list1 = []
            for token in nlp(keyword) :
            #     print({'LOWER' : token.text.lower()})
                if token.text == '.' :
                    if len(pattern_list1) > 1:                        
                        pattern_list1[i-1]['LOWER'] = pattern_list1[i-1].get('LOWER') + "."
                        # print(pattern_list1[i-1].get('LOWER') + ".")
                else:
                    pattern_list1.append({'LOWER' : token.text.lower()}) 
                    # print({'LOWER' : token.text.lower()}) 
                i=i+1
            
            if len(keyword.split()) > 1:
                match_id = ''.join(el[0] for el in keyword.split())
            else:
                match_id = keyword

            self.true_matcher.add(match_id, None, pattern_list1)

        for keyword in self.false_keywords :
            i=0
            pattern_list1 = []
            for token in nlp(keyword) :
            #     print({'LOWER' : token.text.lower()})
                if token.text == '.' :
                    if len(pattern_list1) > 1:                        
                        pattern_list1[i-1]['LOWER'] = pattern_list1[i-1].get('LOWER') + "."
                        # print(pattern_list1[i-1].get('LOWER') + ".")
                else:
                    pattern_list1.append({'LOWER' : token.text.lower()})
                    # print({'LOWER' : token.text.lower()}) 
                i=i+1
            
            if len(keyword.split()) > 1:
                match_id = ''.join(el[0] for el in keyword.split())
            else:
                match_id = keyword

            self.false_matcher.add(match_id, None, pattern_list1)

        print("generated patterns...")
        print("time : ", time.time() - start)
        # return self.matcher
            
    def tag_data(self, sents):
        print("tagging data...\n")
        start = time.time()
        total_y_sents = []
        y_sents = []
        sent_indexes = []
        matched_len = 0
        unmatched_len = 0

        # total_y_sents = np.empty(shape=(len(s),2), dtype='object')
        total_y_sents = []
        y_sents = []
        # y_sents = np.empty(shape=(len(s),2), dtype='object')
        for j, s in enumerate(sents) :
            true_matches = self.true_matcher(nlp(str(s)))
            false_matches = self.false_matcher(nlp(str(s)))
            
            s_ = [word.text for word in s]
            s_ = np.array(s_)

            y_sents_temp = np.empty(shape=(len(s),2), dtype='object')

            if true_matches:
                true_list_indexes = []
                for tm in true_matches:
                    true_list_indexes.extend(np.arange(tm[1], tm[2]))
                true_list_indexes = list(sorted(set(true_list_indexes)))
                # print("true list indexes : ", true_list_indexes)

                if false_matches:
                    false_list_indexes = []
                    for fm in false_matches:
                        false_list_indexes.extend(np.arange(fm[1],fm[2]))
                    false_list_indexes = list(sorted(set(false_list_indexes)))
                    # print("false list indexes : ", false_list_indexes)
                        
                
                temp_t= zip(s_[true_list_indexes], ['yes' for i in range(len(true_list_indexes))])
                # print ([t for t in temp_t])
                
                if false_matches:
                    temp_f= zip(s_[false_list_indexes], ['no' for i in range(len(false_list_indexes))])
                    # print ([t for t in temp_f])

                other_mask = np.zeros(s_.shape, dtype=bool)
                other_mask[true_list_indexes] = 1
                if false_matches:
                    other_mask[false_list_indexes] = 1
                    # print(s_[~other_mask])
                    temp_o= zip(s_[~other_mask], ['other' for i in range(len(s_) -(len(false_list_indexes) + len(true_list_indexes)))])
                    temp_o_copy = copy.deepcopy(temp_o)
                    # print("BOTH asserting other mask : ", np.sum(~other_mask), " -- > ", len(s_) -(len(false_list_indexes) + len(true_list_indexes)))
                    y_sents_temp[false_list_indexes] = [t for t in temp_f]
                    # print("BOTH len other ", len(y_sents_temp[~other_mask]))
                    # print("BOTH len other ", y_sents_temp[~other_mask])
                    y_sents_temp[~other_mask] = [t for t in temp_o]
                else:
                    temp_o= zip(s_[~other_mask], ['other' for i in range(len(s_) -len(true_list_indexes))])    
                    # print("NO FALSE len other ", len(y_sents_temp[~other_mask]))
                    # print("NO FALSE y_sents other ", y_sents_temp[~other_mask])
                    # print("NO FALSE asserting other mask : ", np.sum(~other_mask), " -- > ", len(s_) - len(true_list_indexes))
                    # temp_o_copy = copy.deepcopy(temp_o)
                    # print ("temp o copy : ", [t for t in temp_o_copy])
                    if len(y_sents_temp[~other_mask]) > 0:
                        y_sents_temp[~other_mask] = [t for t in temp_o]
                    
                
                # print("sentence working : ", s_)

                y_sents_temp[true_list_indexes] = [t for t in temp_t]
                
                # y_sents_temp[~other_mask] = [t for t in temp_o]

                y_sents.extend(y_sents_temp.tolist())
                total_y_sents.extend(y_sents_temp.tolist())
                sent_indexes.extend([self.total_sent_number])
                self.total_sent_number += 1
            else:                                
                temp_o= zip(s_, ['other' for i in range(len(s_))])
                total_y_sents = [t for t in temp_o]
                self.total_sent_number += 1

            # if (y_sents_temp and sent_indexes):
            print("actual sent :\n", s_)
            print("yaewnts, total, indexes :\n", y_sents, "\n", total_y_sents, "\n", sent_indexes)
            # print("asserting -- > ", len(y_sents_temp), len(sent_indexes))


        # self.total_sent_number += len(total_y_sents)
        self.sents.extend(sents)
        self.y_sents.extend(y_sents)
        self.total_y_sents.extend(total_y_sents)
        self.sent_indexes.extend(sent_indexes)
        print("tagged data...")
        print("time : ", time.time() - start)

        return y_sents, sent_indexes, total_y_sents     

    def write_to_pickle(self, drug, sents_list, y_sents, total_y_sents, sent_indexes):
        sents_dict = dict([(drug, s) for s in sents_list])
        y_sents_dict = dict([(drug, y) for y in y_sents])
        total_y_sents_dict = dict([(drug, t) for t in total_y_sents])
        sent_indexes_dict = dict([(drug, i) for i in sent_indexes])

        print("sents dict : \n", sents_dict)
        print("y_sents_dict:\n", y_sents_dict)
        print("total_y_sents_dict:\n", total_y_sents_dict)
        print("sents_indexes_dict:\n", sent_indexes_dict)

        with open(DATA_STORE + 'sents.pkl', 'ab') as f:
            
            pickle.dump(sents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(DATA_STORE + 'y_sents.pkl', 'ab') as f:
            
            pickle.dump(y_sents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(DATA_STORE + 'total_y_sents.pkl', 'ab') as f:
            
            pickle.dump(total_y_sents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(DATA_STORE + 'sent_indexes.pkl', 'ab') as f:
            
            pickle.dump(sent_indexes_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            

    def read_from_pickle(self):
        with open(DATA_STORE + 'sents.pkl', 'rb', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.sents.append(json.loads(row))

        with open(DATA_STORE + 'y_sents.pkl', 'rb', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.y_sents.append(json.loads(row))

        with open(DATA_STORE + 'total_y_sents.pkl', 'rb', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.total_y_sents.append(json.loads(row))

        with open(DATA_STORE + 'sent_indexes.pkl', 'rb', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.sent_indexes.append(json.loads(row))

    def word2feat1(self, sent, i, feature_select = 0):
        
        word = sent[i]
    #     word = nlp(word)
# ---------------------------------------
        # feature 1
        feature1 = {
            'bias' : 1.0,
            'pos' : word.pos_,
            'word[-1:]' : word.text[-1:],
            # 'word[-2:]' : word.text[-2:],
            # 'word[-3:]' : word.text[-3:],
            'word[:1]' : word.text[:1],
            # 'word[:2]' : word.text[:2],
            # 'word[:3]' : word.text[:3],
            'word.lower()' : str(word).lower(),
            'word.isdigit()' : str(word).isdigit(),
            'word.is_alpha()' : word.is_alpha,
            'word.is_stop()' : word.is_stop,
            'word.istitle()' : word.text[0].upper() == word.text[0],
            'word.is_special_char()' : not(re.findall("[A-Za-z0-9]", word.text)),
            # 'head' : word.head,
            'head_tag' : word.head.tag_,
            'head_pos' : word.head.pos_,
            'head_dep' : word.head.dep_,
            'dep' : word.dep_,
            'tag' : word.tag_,
            'shape' : word.shape_
        }
        # i=0
        # for l in word.lefts:            
        #     feature1.update({
        #         'lefts_' + str(i) : l.text,
        #         'lefts_pos' + str(i) : l.pos_            
        #     })
        #     i+=1
        # i=0
        # for r in word.rights:            
        #     feature1.update({
        #         'rights_' + str(i) : r.text,
        #         'rights_pos' + str(i) : r.pos_            
        #     })
        #     i+=1
        # for num in range(len(self.keywords)):
        #     feature1.update({'fuzzy_distance_' + str(num) : fuzz.partial_ratio(word.text.lower(), self.keywords[num])})  # should be vera level feature set        
    
        if i > 0:
            word1 = sent[i-1]
            feature1.update(
                {
            '-1:pos' : word1.pos_,
    #         '-1:word[-1:]' : word1.text[-1:],
    #         '-1:word[-2:]' : word1.text[-2:],
    #         '-1:word[-3:]' : word1.text[-3:],
    #         '-1:word.lower()' : str(word1).lower(),
            '-1:word.isdigit()' : str(word1).isdigit(),
            '-1:word.is_alpha()' : word1.is_alpha,
            '-1:word.is_stop()' : word1.is_stop,
            # '-1:bigram.word()' : word1.text + ' ' + word.text,
            '-1:bigram.tag()' : word1.tag_ + ' ' +  word.tag_  ,              
            '-1:bigram.dep()' : word1.dep_ + ' ' + word.dep_   ,             
            '-1:dep' : word1.dep_,
            '-1:tag' : word1.tag_,
            '-1:shape' : word1.shape_
            }
            )
            # i=0
            # for l in word1.lefts:            
            #     feature1.update({
            #         '-1:lefts_' + str(i) : l.text,
            #         '-1:lefts_pos' + str(i) : l.pos_            
            #     })
            #     i+=1
            # i=0
            # for r in word1.rights:            
            #     feature1.update({
            #         '-1:rights_' + str(i) : r.text,
            #         '-1:rights_pos' + str(i) : r.pos_            
            #     })
            #     i+=1
            
        else:
            feature1['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1]
            feature1.update({
            '+1:pos' : word1.pos_,
    #         '+1:word[-1:]' : word.text[-1:],
    #         '+1:word[-2:]' : word.text[-2:],
    #         '+1:word[-3:]' : word.text[-3:],
    #         '+1:word.lower()' : str(word1).lower(),
            '+1:word.isdigit()' : str(word1).isdigit(),
            '+1:word.is_alpha()' : word1.is_alpha,
            '+1:word.is_stop()' : word1.is_stop,
            # '+1:bigram.word()' : word.text + word1.text,
            '+1:bigram.tag()' : word.tag_ + word1.tag_  ,              
            '+1:bigram.dep()' : word.dep_ + word1.dep_ ,

            '+1:dep' : word1.dep_,
            '+1:tag' : word1.tag_,
            '+1:shape' : word1.shape_
            })
        else:
            feature1['EOS'] = True            
# -----------------------------------
# ------------------------------------
        # feature 2
        feature2 = {
                'bias' : 1.0,
                'pos' : word.pos_,
                'word[-1:]' : word.text[-1:],
                # 'word[-2:]' : word.text[-2:],
                # 'word[-3:]' : word.text[-3:],
                'word[:1]' : word.text[:1],
                # 'word[:2]' : word.text[:2],
                # 'word[:3]' : word.text[:3],
                # 'word.lower()' : str(word).lower(),
                'word.abbrev' : "".join(el[0] if '_' in word.text else el for el in word.text.split('_') ),
                # 'word.isabbrev()' : ,
        #         'word.isdigit()' : str(word).isdigit(),
                'word.is_alpha()' : word.is_alpha,
        #         'word.is_stop()' : word.is_stop,
                'word.istitle()' : word.text[0].upper() == word.text[0],
                
        #         'dep' : word.dep_,
                'tag' : word.tag_,
        #         'shape' : word.shape_
            }
        
        # for num in range(len(self.keywords_no_spaces)):
        #     feature2.update({'fuzzy_distance' : fuzz.partial_ratio(word.text.lower(), "".join(el[0] for el in self.keywords[num].split('_')))})


        if i > 0:
            word1 = sent[i-1]
            feature2.update(
                {
            '-1:pos' : word1.pos_,
    #         '-1:word[-1:]' : word1.text[-1:],
    #         '-1:word[-2:]' : word1.text[-2:],
    #         '-1:word[-3:]' : word1.text[-3:],
    #         '-1:word.lower()' : str(word1).lower(),
            '-1:word.isdigit()' : str(word1).isdigit(),
            '-1:word.is_alpha()' : word1.is_alpha,
            '-1:word.is_stop()' : word1.is_stop,
            # '-1:bigram.word()' : word1.text + ' ' + word.text,
            '-1:bigram.tag()' : word1.tag_ + ' ' +  word.tag_  ,              
            '-1:bigram.dep()' : word1.dep_ + ' ' + word.dep_   ,             
            '-1:dep' : word1.dep_,
            '-1:tag' : word1.tag_,
            '-1:shape' : word1.shape_
            }
            )
            
        else:
            feature2['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1]
            feature2.update({
            '+1:pos' : word1.pos_,
    #         '+1:word[-1:]' : word.text[-1:],
    #         '+1:word[-2:]' : word.text[-2:],
    #         '+1:word[-3:]' : word.text[-3:],
    #         '+1:word.lower()' : str(word1).lower(),
            '+1:word.isdigit()' : str(word1).isdigit(),
            '+1:word.is_alpha()' : word1.is_alpha,
            '+1:word.is_stop()' : word1.is_stop,
            # '+1:bigram.word()' : word.text + word1.text,
            '+1:bigram.tag()' : word.tag_ + word1.tag_  ,              
            '+1:bigram.dep()' : word.dep_ + word1.dep_ ,
            '+1:dep' : word1.dep_,
            '+1:tag' : word1.tag_,
            '+1:shape' : word1.shape_
            })
        else:
            feature2['EOS'] = True
# ------------------------------------

        features = [feature1, feature2]

        return features[feature_select]

    def sent2labels(self, sent):        
        return [label for text, label in sent]

    def sent2features(self, sent, feature_select):
        
        return [self.word2feat1(sent, i, feature_select) for i in range(len(sent))]


    def create_train_data(self, sents = None, y_sents=None, sent_indexes = None, total_y_sents = None, feature_select = 0):
        print("Creating training data...")
        
        start = time.time()
        # self.write_to_csv()

        if not sents:
            sents = self.sents
        if not y_sents:            
            y_sents=self.y_sents            
        if not sent_indexes:    
            sent_indexes = self.sent_indexes
        if not total_y_sents:    
            total_y_sents = self.total_y_sents

        print(sent_indexes)
        print(len(y_sents))
        print(len(total_y_sents))
        # print(type(sents))
        # print(type(y_sents))
        # print(type(sent_indexes))
        # print(type(total_y_sents))

        r = np.random.randint(0, len(total_y_sents))
        print("rand: ", r)
        print([i for i in sent_indexes])

        # X_train = [self.sent2features(s, feature_select) for s in [sents[i] for i in sent_indexes]] + [self.sent2features(sents[r], feature_select)] +[self.sent2features(sents[-1], feature_select)] + [self.sent2features(sents[-2], feature_select)]
        # y_train = [self.sent2labels(s) for s in y_sents] + [self.sent2labels(total_y_sents[r])] + [self.sent2labels(total_y_sents[-1])] + [self.sent2labels(total_y_sents[-2])]

        # X_test = [self.sent2features(s, feature_select) for s in [sents[i] for i in sent_indexes]] + [self.sent2features(sents[r], feature_select)] + [self.sent2features(sents[-1], feature_select)] + [self.sent2features(sents[-2], feature_select)]
        # y_test = [self.sent2labels(s) for s in y_sents] + [self.sent2labels(total_y_sents[r])] + [self.sent2labels(total_y_sents[-1])] + [self.sent2labels(total_y_sents[-2])]

        X_train_full = [self.sent2features(s, feature_select) for s in sents]
        y_train_full = [self.sent2labels(s) for s in total_y_sents]
        X_train, X_test = train_test_split(X_train_full, test_size=0.2)
        y_train, y_test = train_test_split(y_train_full, test_size=0.2)

        print("Created training data...")
        
        print("len X: ", len(X_train))
        print("len y: ", len(y_train))
        # print("len X: ", len(X_train[0]))
        # print("len y: ", len(y_train[0]))
        temp_train = []
        temp_test = []
        # for i in range(len(sent_indexes)):
        #     print(i)
        #     print(len(sents[sent_indexes[i]]))
        #     print(len(y_test[i]))
        #     print(sents[sent_indexes[i]])
        #     print(y_test[i])
        
        for i in range(len(X_train)):
            # if i == 0:
            #     temp =i
            #     continue
            # print(i)
            # print(self.y_sents[i])
            # print(self.sent_indexes[i])
            # print("len sents", len(self.y_sents[i]))
            # print("len total y sents", len(self.sent_indexes[i]))
            # print("X train : ", len(X_train[i]))
            
            # print("X infer : ", [X_infer[i][j].get('word.abbrev') for j in range(len(X_infer[i]))])
            # print("y train : ", len(y_train[i]))
            # print(y_train[i])
            # print("y true : ", y_true[i])
            try:
                assert len(X_train[i]) == len(y_train[i])
            except:
                print("skipping train ssertion...")
                temp_train.append(i)

            try:
                assert len(X_test[i]) == len(y_test[i])
            except:
                print("skipping test ssertion...")
                temp_test.append(i)
            # print("asserted")
        for index in sorted(temp_train, reverse=True):
            del X_train[index]
            del y_train[index]

        for index in sorted(temp_test, reverse=True):
            del X_test[index]
            del y_test[index]

        
        
        print("sent indexes : ", len(sent_indexes))
        print("y_sent : ", len(y_sents))
        print("len X_Train : ", len(X_train))
        print("len y_train : ", len(y_train))
        print("len X_test : ", len(X_test))
        print("len y_test : ", len(y_test))

        print("time : ", time.time() - start)

        return X_train, y_train, X_test, y_test

    def create_inference_data(self, sents = None, total_y_sents = None, feature_select = 0):
        start = time.time()
        if not sents:
            sents = self.sents
        # if not y_sents:            
        #     y_sents=self.y_sents            
        # if not sent_indexes:    
        #     sent_indexes = self.sent_indexes
        print("sents and y_total_sents : \n", len(self.sents), len(self.total_y_sents))
        if not total_y_sents:    
            total_y_sents = self.total_y_sents

        y_true = [self.sent2labels(s) for s in total_y_sents]
        X_infer = [self.sent2features(s, feature_select) for s in sents]

        print("lens of x and y : \n", len(y_true), len(X_infer))

        temp =[]
        for i in range(len(y_true)):

            # print(i)
            # print(self.sents[i])
            # print(self.total_y_sents[i])
            # print("len sents", len(self.sents[i]))
            # print("len total y sents", len(self.total_y_sents[i]))
            # print("X infer : ", len(X_infer[i]))
            # print("X infer : ", [X_infer[i][j].get('word.abbrev') for j in range(len(X_infer[i]))])
            # print("y true : ", len(y_true[i]))
            # print("y true : ", y_true[i])
            
            try:
                assert len(X_infer[i]) == len(y_true[i])
            except:
                print("skipping ssertion...")
                temp.append(i)
            # print("asserted")
        for index in sorted(temp, reverse=True):
            del X_infer[index]
            del y_true[index]
        print("len of y true : ", len(y_true))
        print("len of X infer : ", len(X_infer))
        print("created infernce data...")
        print("time : ", time.time() - start)

        return X_infer, y_true, sents

def main():
    print("Streaming data pipeline started...")

    stream = Data_Stream()
    start = time.time()
    # _ = stream.gen_pattern(keywords = None)
    # print("time : ", time.time() - start)
    # start = time.time()
    # _, _, _ = stream.tag_data()
    # print("time : ", time.time() - start)
    # start = time.time()
    X_train, y_train, X_test, y_test = stream.create_train_data( feature_select=0)
    print("time : ", time.time() - start)
    start = time.time()
    X_infer, y_true, sents = stream.create_inference_data( feature_select = 0)
    print("time : ", time.time() - start)

    # m = stream.gen_pattern_no_spaces(keywords = None)
    # sents, sents_no_spaces, y_sents, sent_indexes, total_y_sents  = stream.tag_data_no_spaces()
    # X_train, y_train, X_test, y_test = stream.create_train_data(sents_no_spaces, y_sents, sent_indexes, total_y_sents, feature_select=1)
    # X_infer, y_true, sents = stream.create_inference_data(sents, total_y_sents, feature_select = 1)

if __name__ == '__main__':
    main()