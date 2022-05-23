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
from flashtext import KeywordProcessor
import multiprocessing as mpfull
from multiprocessing import get_context 


nlp = spacy.load('en')

DATA_INP = '/Users/ashwins/repo/DataScience/dailymed_crf/data/'
DATA_STORE = '/Users/ashwins/repo/DataScience/dailymed_crf/train_data/'

data_df = pd.read_csv(DATA_INP + 'input_crf_new.csv', header=None)
data_df = data_df.iloc[:50]
df_search = pd.read_csv('/Users/ashwins/Scripts/dd_map_scrape/done_data/consolidated_final.csv', encoding='utf-8', index_col=0)

def split_handler(df):
    sents_split = []
    true_x_sents_split = []
    true_y_sents_split = []
    false_x_sents_split = []
    false_y_sents_split = []
    y_sents_split = []
    sent_indexes_split = []
    total_y_sents_split = []

    for i, row in df.iterrows():
        sents_temp, y_sents_temp, true_x_sents_temp, true_y_sents_temp, false_x_sents_temp, false_y_sents_temp, total_y_sents_temp = sequence_handler(row)
        sents_split.extend(sents_temp)
        y_sents_split.extend(y_sents_temp)
        true_x_sents_split.extend(true_x_sents_temp)
        true_y_sents_split.extend(true_y_sents_temp)
        false_x_sents_split.extend(false_x_sents_temp)
        false_y_sents_split.extend(false_y_sents_temp)
        total_y_sents_split.extend(total_y_sents_temp)

    try:

        print("false_x_sents[0] : ", false_x_sents_split[0])
        print("false_y_sents[0] : ", false_y_sents_split[0])
        print("sent_indexes_split[0] : ", sent_indexes_split[0])
        print("put in queue")
    except:
        print("none put in queue")

    return [json.dumps(sents_split),json.dumps(y_sents_split),json.dumps(total_y_sents_split), json.dumps(true_x_sents_split), json.dumps(true_y_sents_split),json.dumps(false_x_sents_split), json.dumps(false_y_sents_split)]


def sequence_handler( row):
    true_matcher, false_matcher, true_keywords, false_keywords = gen_pattern(row[1])
    
    para = re.sub(r"\(|\)", "", row[3])
    sents = [s for  s in nlp(re.sub(r"\n|u\'|•|\\xa0", " ", para)).sents]
    sents_list = [[word.text for word in s] for s in sents]
    
    true_x_sents_seq, true_y_sents_seq, false_x_sents_seq, false_y_sents_seq, y_sents_seq, total_y_sents_seq = tag_data(row[1] ,sents, true_matcher, false_matcher)

    return sents_list, y_sents_seq, true_x_sents_seq, true_y_sents_seq, false_x_sents_seq, false_y_sents_seq, total_y_sents_seq

    
def get_keywords( drug_name):
    
    false_keywords = df_search[(df_search['drug'] == drug_name) & (df_search['type'] == 'n')]['indication'].str.lower().astype('str').tolist()
    true_keywords = df_search[(df_search['drug'] == drug_name) & (df_search['type'] == 'y')]['indication'].str.lower().astype('str').tolist()

    return true_keywords, false_keywords

def gen_pattern( drug_name):

    print("generating patterns... \n")
    start = time.time()
    true_matcher = Matcher(nlp.vocab)
    false_matcher = Matcher(nlp.vocab)

    true_keywords, false_keywords = get_keywords(drug_name)

    for keyword in true_keywords :
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

        true_matcher.add(match_id, None, pattern_list1)

    for keyword in false_keywords :
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

        false_matcher.add(match_id, None, pattern_list1)

    print("generated patterns...")
    print("time : ", time.time() - start)
    return true_matcher, false_matcher, true_keywords, false_keywords

        
def tag_data(drug_name, sents, true_matcher, false_matcher):
    print("\ntagging data...")
    start = time.time()
    total_y_sents = []
    y_sents = []
    true_x_sents = []
    true_y_sents = []
    false_x_sents = []
    false_y_sents = []
    sent_indexes = []
    matched_len = 0
    unmatched_len = 0

    total_sent_number = 0


    for j, s in enumerate(sents) :
        true_matches = true_matcher(nlp(str(s)))
        false_matches = false_matcher(nlp(str(s)))
        
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
                # temp_o_copy = copy.deepcopy(temp_o)
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

            
            y_sents_temp[true_list_indexes] = [t for t in temp_t]                            

            true_x_sents.append(s_.tolist())
            true_y_sents.append(y_sents_temp.tolist())
            y_sents.append(y_sents_temp.tolist())
            total_y_sents.append(y_sents_temp.tolist())
            sent_indexes.extend([total_sent_number])
            total_sent_number += 1
        
        elif false_matches:      
            false_list_indexes = []
            other_mask = np.zeros(s_.shape, dtype=bool)
            for fm in false_matches:
                false_list_indexes.extend(np.arange(fm[1],fm[2]))
            false_list_indexes = list(sorted(set(false_list_indexes)))

            temp_f= zip(s_[false_list_indexes], ['no' for i in range(len(false_list_indexes))])
            other_mask[false_list_indexes] = 1
            
            temp_o= zip(s_[~other_mask], ['other' for i in range(len(s_) -(len(false_list_indexes)))])
            # temp_o_copy = copy.deepcopy(temp_o)
            # print("BOTH asserting other mask : ", np.sum(~other_mask), " -- > ", len(s_) -(len(false_list_indexes) + len(true_list_indexes)))
            y_sents_temp[false_list_indexes] = [t for t in temp_f]
            # print("BOTH len other ", len(y_sents_temp[~other_mask]))
            # print("BOTH len other ", y_sents_temp[~other_mask])
            if len(y_sents_temp[~other_mask]) > 0:
                y_sents_temp[~other_mask] = [t for t in temp_o]

            # y_sents_temp[false_list_indexes] = [t for t in temp_f]                            
            false_x_sents.append(s_.tolist())
            false_y_sents.append(y_sents_temp.tolist())
            y_sents.append(y_sents_temp.tolist())
            sent_indexes.extend([total_sent_number])
            total_y_sents.append(y_sents_temp.tolist())
            total_sent_number += 1

        else:

            temp_o= zip(s_, ['other' for i in range(len(s_))])
            # temp_o_copy = copy.deepcopy(temp_o)
            # print("\ntemp_o_copy in not found...:", [t for t in temp_o_copy], "\n\n")
            total_y_sents.append([t for t in temp_o])
            total_sent_number += 1


    print("tagged data...")
    print("time : ", time.time() - start)

    return true_x_sents, true_y_sents, false_x_sents, false_y_sents, y_sents, total_y_sents 


class Data_Stream():
    
    def __init__(self, file_path = DATA_INP):
        self.data = ""        
        self.data_df = pd.read_csv(DATA_INP + 'input_crf_new.csv', header=None)
        pd.set_option('max_columns' , 20)
        print(self.data_df.head())
        self.sents = []
        self.false_x_sents = []
        self.false_y_sents = []
        self.y_sents= []
        self.sent_indexes = []
        self.total_y_sents = []
        self.total_sent_number = 0

        self.sents_list = []
        self.sents_tuple = []
        self.y_sents_tuple = []
        self.total_y_sents_tuple = []
        self.sent_indexes_tuple = []

        self.df_search = pd.read_csv('/Users/ashwins/Scripts/dd_map_scrape/done_data/consolidated_final.csv', encoding='utf-8', index_col=0)

    def write_to_pickle_try(self):
        
        with open(DATA_STORE + 'sents_try.pkl', 'ab') as f:            
            pickle.dump(self.sents_tuple, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(DATA_STORE + 'y_sents_try.pkl', 'ab') as f:                
            pickle.dump(self.y_sents_tuple, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(DATA_STORE + 'total_y_sents_try.pkl', 'ab') as f:                
            pickle.dump(self.total_y_sents_tuple, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(DATA_STORE + 'sent_indexes_try.pkl', 'ab') as f:                
            pickle.dump(self.sent_indexes_tuple, f, protocol=pickle.HIGHEST_PROTOCOL)


    def write_to_pickle(self, drug, sents, y_sents, total_y_sents, sent_indexes):
        
        sents_list = [[word.text for word in s] for s in self.sents]

        if not len(sents_list) ==0:
            sents_dict = [(drug, s) for s in sents_list]

        try : 
            assert len(y_sents) != 0
            assert len(y_sents[0]) >1             
            y_sents_dict = [(drug, y) for y in self.y_sents]
        except:
            if len(y_sents)!=0:                
                y_sents_dict = [(drug, y_sents)]
        
        try:
            assert len(total_y_sents) !=0
            assert len(total_y_sents[0]) >1
            total_y_sents_dict = [(drug, t) for t in self.total_y_sents]
        except:
            if len(total_y_sents) !=0:
                total_y_sents_dict = [(drug, self.total_y_sents)]
        
        if not len(sent_indexes)==0 :
            sent_indexes_dict = [(drug, i) for i in sent_indexes]
        
        try:
            # print("sents dict : \n", sents_dict)
            # print("y_sents_dict:\n", y_sents_dict)
            # print("total_y_sents_dict:\n", total_y_sents_dict)
            # print("sents_indexes_dict:\n", sent_indexes_dict)

            with open(DATA_STORE + 'sents_temp.pkl', 'ab') as f:            
                pickle.dump(sents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(DATA_STORE + 'y_sents_temp.pkl', 'ab') as f:                
                pickle.dump(y_sents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(DATA_STORE + 'total_y_sents_temp.pkl', 'ab') as f:                
                pickle.dump(total_y_sents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(DATA_STORE + 'sent_indexes_temp.pkl', 'ab') as f:                
                pickle.dump(sent_indexes_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        except:
            print("Empty")
            

    def read_from_pickle(self):
        self.sents = []
        self.y_sents = []
        self.total_y_sents = []
        self.sent_indexes = []
        with open(DATA_STORE + 'sents_try.pkl', 'rb') as f:
            sents = []
            for i in range(430):
            # while 1:
                try:
                    sents.append(pickle.load(f, encoding='utf-8'))
                except EOFError:
                    break

        with open(DATA_STORE + 'y_sents_try.pkl', 'rb') as f:
            y_sents = []
            for i in range(430):
            # while 1:
                try:
                    y_sents.append(pickle.load(f, encoding='utf-8'))
                except EOFError:
                    break

        with open(DATA_STORE + 'total_y_sents_try.pkl', 'rb') as f:
            total_y_sents = []
            for i in range(430):
            # while 1:
                try:
                    total_y_sents.append(pickle.load(f, encoding='utf-8'))
                except EOFError:
                    break

        with open(DATA_STORE + 'sent_indexes_try.pkl', 'rb') as f:
            sent_indexes = []
            for i in range(430):
            # while 1:
                try:
                    sent_indexes.append( pickle.load(f, encoding='utf-8'))
                except EOFError:
                    break

        for t in total_y_sents:   
            if t:
                self.total_y_sents.extend(t) 
        for s in sents:
            if s:
                self.sents.extend(s)
        for y in y_sents:
            if y:
                self.y_sents.extend(y)
        for i in sent_indexes:
            if i:
                self.sent_indexes.extend(i)

        print("len(total_y_sents) during read : ", len(total_y_sents))
        print("len(sents) duing read : ", len(sents))



    def word2feat1(self, sent, i, feature_select = 0):
        
        word = sent[i]
# ---------------------------------------
        # feature 1
        feature1 = {
            'bias' : 1.0,
            'pos' : word.pos_,
            'word[-1:]' : word.text[-1:],
            'word[:1]' : word.text[:1],
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
        
        # for num in range(len(self.true_keywords)):
        #     feature1.update({'fuzzy_distance_' + str(num) : fuzz.partial_ratio(word.text.lower(), self.keywords[num])})  # should be vera level feature set    

        if i>1:
            word1 = sent[i-2]
            feature1.update({
                '-2:negation' : fuzz.ratio(word1.text, 'not')
            })
        if i>2:
            word1 = sent[i-3]
            feature1.update({
                '-3:negation' : fuzz.ratio(word1.text, 'not')
            })
        if i>3:
            word1 = sent[i-4]
            feature1.update({
                '-4:negation' : fuzz.ratio(word1.text, 'not')
            })
    
        if i > 0:
            word1 = sent[i-1]
            feature1.update(
                {
            '-1:pos' : word1.pos_,    
            '-1:word.lower()' : str(word1).lower(),
            '-1:word.isdigit()' : str(word1).isdigit(),
            '-1:word_negation' : fuzz.ratio(word1.text, 'not'),
            '-1:word.is_alpha()' : word1.is_alpha,
            '-1:word.is_stop()' : word1.is_stop,
            
            '-1:bigram.tag()' : word1.tag_ + ' ' +  word.tag_  ,              
            '-1:bigram.dep()' : word1.dep_ + ' ' + word.dep_   ,             
            '-1:dep' : word1.dep_,
            '-1:tag' : word1.tag_,
            '-1:shape' : word1.shape_
            }
            )
            
        else:
            feature1['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1]
            feature1.update({
            '+1:pos' : word1.pos_,    
            '+1:word.lower()' : str(word1).lower(),
            '+1:word.isdigit()' : str(word1).isdigit(),
            '+1:word_negation' : fuzz.ratio(word1.text, 'not'),
            '+1:word.is_alpha()' : word1.is_alpha,
            '+1:word.is_stop()' : word1.is_stop,
            
            '+1:bigram.tag()' : word.tag_ + word1.tag_  ,              
            '+1:bigram.dep()' : word.dep_ + word1.dep_ ,

            '+1:dep' : word1.dep_,
            '+1:tag' : word1.tag_,
            '+1:shape' : word1.shape_
            })
        else:
            feature1['EOS'] = True            
# ------------------------------------
        feature2 = {}
# ------------------------------------
        features = [feature1, feature2]

        return features[feature_select]

    def sent2labels(self, sent):        
        return [label for text, label in sent]

    def sent2features(self, sent, feature_select):
        return [self.word2feat1(sent, i, feature_select) for i in range(len(sent))]

    def create_train_data(self, sents = None, y_sents=None, total_y_sents = None, true_x_sents=None, true_y_sents=None, false_x_sents = None, false_y_sents = None, feature_select = 0):
        
        print("Creating training data...")
        
        start = time.time()

        if not sents:
            sents = self.sents
        if not y_sents:            
            y_sents=self.y_sents            
        if not false_x_sents:    
            false_x_sents = self.false_x_sents
        if not false_y_sents:    
            false_y_sents = self.false_y_sents
        if not total_y_sents:    
            total_y_sents = self.total_y_sents

        sents = [nlp(" ".join(s)) for s in sents]
        print(total_y_sents[:2])
        print(sents[:2])
        # print("len(y_sents) : ", len(y_sents))
        print("len(total_y_sents) : ", len(total_y_sents))
        print("len(sents)", len(sents))
 
        # print(type(sents))
        # print(type(y_sents))
        # print(type(sent_indexes))
        # print(type(total_y_sents))
        # print([i for i in sent_indexes])
        
        # get false sentences..
        X_train_false = [self.sent2features(nlp(" ".join(fx)), feature_select) for fx in false_x_sents]
        y_train_false = [self.sent2labels(fy) for fy in false_y_sents]
        print("len(X_train_false) : ", len(X_train_false))
        print("len(y_train_false) : ", len(y_train_false))
        # get true sentences..
        X_train_true = [self.sent2features(nlp(" ".join(tx)), feature_select) for tx in true_x_sents]
        y_train_true = [self.sent2labels(ty) for ty in true_y_sents]
        print("len(X_train_true) : ", len(X_train_true))
        print("len(y_train_true) : ", len(y_train_true))
        X_train_split_correct, X_test_split_correct, y_train_split_correct, y_test_split_correct = train_test_split(X_train_false+X_train_false+X_train_false+X_train_true, y_train_false+y_train_false+y_train_false+y_train_true, test_size=0.2, random_state=0)
        _, test_sents_correct = train_test_split(false_x_sents+false_x_sents+false_x_sents+true_x_sents, test_size=0.2, random_state=0)
        # get other sents...
        X_train_other = [self.sent2features(s, feature_select) for s in sents]
        y_train_other = [self.sent2labels(t) for t in total_y_sents]
        print("len(X_train_other) : ", len(X_train_other))
        print("len(y_train_other) : ", len(y_train_other))

        X_train_split_other, X_test_split_other, y_train_split_other, y_test_split_other = train_test_split(X_train_other, y_train_other, test_size=0.2, random_state=0)
        _, test_sents_other = train_test_split(sents, test_size=0.2, random_state=0)

        X_train = X_train_split_correct + X_train_split_other
        y_train = y_train_split_correct + y_train_split_other
        X_test = X_test_split_correct + X_test_split_other
        y_test = y_test_split_correct + y_test_split_other

        test_sents = test_sents_correct + test_sents_other

        print("Created training data...")
        
        print("len X: ", len(X_train))
        print("len y: ", len(y_train))
        print("len X: ", len(X_test))
        print("len y: ", len(y_test))
        
        temp_train = []
        temp_test = []
        
        for i in range(len(X_train)):
            
            # print("X infer : ", [X_infer[i][j].get('word.abbrev') for j in range(len(X_infer[i]))])
            # print("y train : ", len(y_train[i]))
            # print(y_train[i])

            try:
                assert len(X_train[i]) == len(y_train[i])
            except:
                print("skipping train ssertion...")
                temp_train.append(i)

            if i < len(X_test):
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
        
        # print("sent indexes : ", len(sent_indexes))
        print("y_sent : ", len(y_sents))
        print("len X_Train : ", len(X_train))
        print("len y_train : ", len(y_train))
        print("len X_test : ", len(X_test))
        print("len y_test : ", len(y_test))

        print("time for training data gen : ", time.time() - start)

        return X_train, y_train, X_test, y_test, test_sents

    def create_inference_data(self, sents, total_y_sents, feature_select = 0):
        start = time.time()
        # sents = [nlp(" ".join(s)) for s in sents]
    
        print("sents and y_total_sents : \n", len(sents), len(total_y_sents))
        y_true = [self.sent2labels(s) for s in total_y_sents]
        X_infer = [self.sent2features((nlp(" ".join(s))), feature_select) for s in sents]

        print("lens of x and y : \n", len(y_true), len(X_infer))

        temp =[]
        for i in range(len(y_true)):
            
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


def load_sents_multiprocessing():

    sents = []
    y_sents = []
    total_y_sents = []
    true_x_sents = []
    true_y_sents = []
    false_x_sents = []
    false_y_sents = []

    mp = mpfull.get_context("spawn")
    split_dfs = np.array_split(data_df, 4)
    with mp.Pool(processes=4) as p :
        results =  p.map(split_handler, split_dfs, chunksize=1)

    for i in range(4):
        print("getting results: ", type(results))
        
        # print("getting sents: ", json.loads(results[0][0])[0])
        # print("getting y_sents: ", json.loads(results[0][1])[0])
        # print("getting total_y_sents: ", json.loads(results[0][2])[0])
        # print("getting false_x_sents: ", json.loads(results[0][3])[0])
        # print("getting false_y_sents: ", json.loads(results[0][4])[0])

        sents.extend(json.loads(results[0][0]))
        y_sents.extend(json.loads(results[0][1]))
        total_y_sents.extend(json.loads(results[0][2]))
        true_x_sents.extend(json.loads(results[0][3]))
        true_y_sents.extend(json.loads(results[0][4]))
        false_x_sents.extend(json.loads(results[0][5]))
        false_y_sents.extend(json.loads(results[0][6]))

    print("len of sents", len(sents))
    print("len of y_sents", len(y_sents))
    print("len of total_y_sents", len(total_y_sents))
    print("len of true_x_sents", len(true_x_sents))
    print("len of true_y_sents", len(true_y_sents))
    print("len of false_x_sents", len(false_x_sents))
    print("len of false_y_sents", len(false_y_sents))

    print("Loaded doc and sents...")
    return sents, y_sents, total_y_sents, true_x_sents, true_y_sents, false_x_sents, false_y_sents

def main():
    init_time= time.time()
    print("Streaming data pipeline started...")

    stream = Data_Stream()
    print ("Stream Over")
    start = time.time()
    sents, y_sents, total_y_sents, true_x_sents, true_y_sents, false_x_sents, false_y_sents = load_sents_multiprocessing()
    print("time : ", time.time() - start)
    start = time.time()
    X_train, y_train, X_test, y_test, test_sents = stream.create_train_data(sents, y_sents, total_y_sents,false_x_sents=false_x_sents, false_y_sents=false_y_sents, true_x_sents=true_x_sents, true_y_sents=true_y_sents)
    print("time : ", time.time() - start)
    start = time.time()
    X_infer, y_true, sents = stream.create_inference_data(sents, total_y_sents, feature_select = 0)
    print("time : ", time.time() - start)
    print("total time : ", time.time() - init_time)

if __name__ == '__main__':
    main()