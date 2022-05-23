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



nlp = spacy.load('en')

DATA_INP = '/Users/ashwins/repo/DataScience/dailymed_crf/data/'
DATA_STORE = '/Users/ashwins/repo/DataScience/dailymed_crf/train_data/'

class Data_Stream():

    def df_sent_creator(self, row):
        self.sents.append([sent for sent in nlp(row[1]).sents])
        print("sentences : \n", [sent for sent in nlp(row[1]).sents])
    
    def __init__(self, file_path = DATA_INP):
        self.data = ""        
        data_df = pd.read_csv(DATA_INP + 'extracted_data.csv', encoding='utf-8', header=None)
        pd.set_option('max_columns' , 20)
        print(data_df)

        # self.doc = nlp(self.data)
        # self.doc = nlp.doc(data_df['STR_Root'].str)
        self.sents = []
        # data_df.apply(lambda x: self.df_sent_creator(x), axis=1)
        data_df.apply(lambda x: self.sents.append([sent for sent in nlp(re.sub(r"\\n|u\'", " ", x[2])).sents]), axis=1) # re.sub(r"[^A-Za-z0-9 ]", " ", 
        # print("sentsLkjchsljdcnl: \n")
        # for s in self.sents:
        #     print(s)
        self.sents = [s for p in self.sents for s in p]
        # self.sents = [sent for sent in self.doc.sents] 
        print("self.sentecnes : \n" ) # ,self.sents      
        self.sents_no_spaces = []
        for s in self.sents:
            tmp_sent = []
            
            for w in s:
            
                tmp_sent.append(w.text)
            self.sents_no_spaces.append(tmp_sent)

        for i,s in enumerate(self.sents):
            assert len(self.sents[i]) == len(self.sents_no_spaces[i])
            # print("asserted...")

        print(len(self.sents))
        print(len(self.sents_no_spaces))
        self.eng = create_engine('mysql+pymysql://sherlock:z00mrxr0cks!@69.164.196.100:3306/ds_data',echo=True)
        # md = MetaData(self.eng)
        print("Loaded doc and sents...")

    def get_keywords(self):

        start = time.time()
        # df_search = pd.read_sql("select * from DrugDisorder_Trie_Details where STY_Root like 'Disorder' limit 100;", self.eng)
        print( "Connected to DB in", time.time() - start, "secs")
        start = time.time()
        # self.keywords = df_search['STR'].tolist()

        self.keywords = ['hypertension', 'Benign Prostatic Hyperplasia', 'dermatoses', 'Heart Failure', 'Myocardial Infarction', 'acne vulgaris', 'Generalized Anxiety Disorder', 'Lower Respiratory Tract Infections', 'Acute Bacterial Otitis Media', 'Sinusitis', 'Skin and Skin Structure Infections', 'Urinary Tract Infections']
        # self.keywords=['leukemia']
        self.keywords_no_spaces = []
        print ("Processed needles and gen keywords in", time.time() - start, "secs")

        for i, kw in enumerate(self.keywords):
            self.keywords_no_spaces.append("_".join(el for el in kw.split()))
        print("time : ", time.time() - start)

    def gen_pattern(self, keywords = None):
        print("generating patterns... \n")
        start = time.time()
        self.matcher = Matcher(nlp.vocab)
        if not keywords :
            self.get_keywords()
        
        # keywords = self.keywords

        for keyword in self.keywords :
            i=0
            pattern_list1 = []
            for token in nlp(keyword) :
            #     print({'LOWER' : token.text.lower()})
                if token.text == '.' :
                    pattern_list1[i-1]['LOWER'] = pattern_list1[i-1].get('LOWER') + "."
                    # print(pattern_list1[i-1].get('LOWER') + ".")
                else:
                    pattern_list1.append({'LOWER' : token.text.lower()}) 
                i=i+1
            
            if len(keyword.split()) > 1:
                match_id = ''.join(el[0] for el in keyword.split())
            else:
                match_id = keyword
        #     print("match_id : ",match_id)
            # print(pattern_list1)
            self.matcher.add(match_id, None, pattern_list1)

        print("generated patterns...")
        print("time : ", time.time() - start)

        return self.matcher
            
    def tag_data(self):
        print("tagging data...\n")
        start = time.time()
        self.processor = KeywordProcessor()
        self.processor.add_keywords_from_list(self.keywords)

        self.total_y_sents = []
        self.y_sents = []
        self.sent_indexes = []
        matched_len = 0
        unmatched_len = 0
        j=0
        for s in self.sents :
        #     doc_sent= nlp(str(s))
            # haystack = str(s)
            # found = list(set(self.processor.extract_keywords(haystack.lower(), span_info=True)))            
            
            # for i, f in enumerate(found):
            #     _sents.append([str(s)[f[1] : f[2]], "disease"])

            matches = self.matcher(nlp(str(s)))
            if matches :
                # print("matched : ", matches)
                _sents = []
                i=0
                for w in s:
                    for match in matches : 
                        if i in range(match[1], match[2]+1) :
                            _sents.append([str(w), "disease"])  #, i]) # w.pos_
                            # rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
                        else : 
                            _sents.append([str(w), "other"]) #, i]) # w.pos_
                    i+=1
                
                self.y_sents.append(_sents)
                self.total_y_sents.append(_sents)  
                
                self.sent_indexes.append(j)
                matched_len += 1
            else :
                _sents = []
                i=0
                for w in s:
                    _sents.append([str(w), "other"]) #, i]) # w.pos_
                    i+=1
                self.total_y_sents.append(_sents)
                unmatched_len +=1
                
            j = j+1

        print("tagged data...")
        print("time : ", time.time() - start)

        return self.y_sents, self.sent_indexes, self.total_y_sents 

    
# ------------------------------- no spaces ---------------------------------------------
    def gen_pattern_no_spaces(self, keywords = None):
        start = time.time()
        self.gen_pattern()

        self.matcher_no_spaces = Matcher(nlp.vocab)

        if not keywords :
            self.get_keywords()
        
        keywords = self.keywords_no_spaces
        print(keywords)
        print(''.join(el[0] for el in keywords[0].split("_")))

        for keyword in keywords :
            i=0
            pattern_list1 = []
            # for token in nlp(keyword) :
                # print({'LOWER' : token.text.lower()})
                # if token.text == '.' :
                #     pattern_list1[i-1]['LOWER'] = pattern_list1[i-1].get('LOWER') + "."
                #     print(pattern_list1[i-1].get('LOWER') + ".")
                # else:
            pattern_list1.append({'LOWER' : keyword.lower()}) 
            i=i+1

            print(pattern_list1)
            # ************** change and see *****************
            # match_id = ''.join(el[0] for el in keyword.split('_'))
            # ***********************************************
            match_id = keyword
            print("mstch id : " ,match_id)
        #     print("match_id : ",match_id)
            self.matcher_no_spaces.add(match_id, None, pattern_list1)

        print("generated patterns...")
        print("time : ", time.time() - start)

        return self.matcher_no_spaces

    def tag_data_no_spaces(self):
        start = time.time()
        self.total_y_sents = []
        self.y_sents = []
        self.sent_indexes = []
        
        matched_len = 0
        unmatched_len = 0
        j=0
        for s in self.sents :
        #     doc_sent= nlp(str(s))
            matches = self.matcher(nlp(str(s)))
            if matches :
                # print("matched : ", matches)
                _sents = []
                i=0
                while i < len(s):
                    # print("J ----> : ", j)
                    # print("I ----> : ", i)
                    for match in matches : 
                        # print("J : ", j)
                        # print("I : ", i)
                        if i in range(match[1], match[2]) :
                            # print("I before n spaces: ", i)
                            # print("self.sents_no_spaces[j][i] : ", self.sents_no_spaces[j][i])
                            # print("J : ", j)
                            # print("I : ", i)
                            # print(match[1], match[2])
                            self.sents_no_spaces[j][i] = nlp.vocab.strings[match[0]]
                            _sents.append([self.sents_no_spaces[j][i], nlp.vocab.strings[match[0]]])  #, i]) # w.pos_
                            del self.sents_no_spaces[j][i+1 : match[2]]
                            
                            i = i + match[2] - match[1] - 1
                            # print("i", i)
                           
                            # i+=1
                            # continue
                            break
                            # rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
                        else : 
                            # print("i in else : ", i, "str(w)", s[i].text)
                            _sents.append([s[i].text, "no"]) #, i]) # w.pos_
                    i+=1
                
                self.y_sents.append(_sents)
                self.total_y_sents.append(_sents)  
                
                self.sent_indexes.append(j)
                matched_len += 1
                
            else :
                _sents = []
                i=0
                for w in s:
                    _sents.append([s[i].text, "no"]) #, i]) # w.pos_
                    i+=1
                self.total_y_sents.append(_sents)
                unmatched_len +=1
                
            j= j+1
            
            # if j==2:
            #     break

        # print("tagged data...\n\n")
        print(self.sent_indexes)
        print("len sents : ", len(self.sents))
        print("len sents no spaces : ", len(self.sents_no_spaces))
        print("len y sents : ", len(self.y_sents))
        print("len total y sents : ", len(self.total_y_sents), "\n\n")

        

        sents_no_spaces = []
        for s in self.sents_no_spaces:
            tmp_sent = ""
            tmp_sent = " ".join(w for w in s)            
            sents_no_spaces.append(nlp(tmp_sent))

        self.sents_no_spaces = sents_no_spaces

        # for j in range(len(self.sents)):
        #     print("sents : ", self.sents[j])
        #     print("no spaces : ", self.sents_no_spaces[j])

        # print(self.sents[0])
        # print(self.sents_no_spaces[0])
        # print("self.sents : ", self.sents, "\n\n")
        # print("self.sents_no_spaces : ", self.sents_no_spaces, "\n\n")
        # print("self.y_sents : ", self.y_sents, "\n\n")

        for i,s in enumerate(self.sents_no_spaces):
            print(len(self.sents_no_spaces[i]),  "-->", len(self.total_y_sents[i]))
            print(self.sents_no_spaces[i], " ---> ", self.total_y_sents[i])
            assert len(self.sents_no_spaces[i]) == len(self.total_y_sents[i])
            # print("asserted ...")

        print("Done tagging no spaces : ", "\n\n")
        print("time : ", time.time() - start)

        return self.sents, self.sents_no_spaces, self.y_sents, self.sent_indexes, self.total_y_sents 
# ------------------------------- no spaces ---------------------------------------------

    def write_to_csv(self):
        with open(DATA_STORE + 'sents.csv', 'w') as f:
            writer = csv.writer(f)
            for i in range(len(self.sents)):
                writer.writerow(self.sents[i])

        with open(DATA_STORE + 'y_sents.csv', 'w') as f:
            writer = csv.writer(f)
            for i in range(len(self.y_sents)):
                # print(self.y_sents[i])
                # print(json.dumps(self.y_sents[i]))
                writer.writerow(json.dumps(self.y_sents[i]))

        with open(DATA_STORE + 'total_y_sents.csv', 'w') as f:
            writer = csv.writer(f)
            for i in range(len(self.total_y_sents)):
                writer.writerow(json.dumps(self.total_y_sents[i]))

        with open(DATA_STORE + 'sent_indexes.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(map(lambda x : [x], self.sent_indexes))
            # for i in range(len(self.sent_indexes)):
            #     writer.writerow(self.sent_indexes[i])

    def read_from_csv(self):
        with open(DATA_STORE + 'sents', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.sents.append(json.loads(row))

        with open(DATA_STORE + 'y_sents', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.y_sents.append(json.loads(row))

        with open(DATA_STORE + 'total_y_sents', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                self.total_y_sents.append(json.loads(row))

        with open(DATA_STORE + 'sent_indexes', 'r', encoding='utf-8') as f:
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
        # print([str(sent[i])+ '\n' for i in range(len(sent))])
        return [self.word2feat1(sent, i, feature_select) for i in range(len(sent))]


    def create_train_data(self, sents = None, y_sents=None, sent_indexes = None, total_y_sents = None, feature_select = 0):
        print("Creating training data...")
        
        start = time.time()
        self.write_to_csv()

        if not sents:
            sents = self.sents
        if not y_sents:            
            y_sents=self.y_sents            
        if not sent_indexes:    
            sent_indexes = self.sent_indexes
        if not total_y_sents:    
            total_y_sents = self.total_y_sents

        # print(type(sents))
        # print(type(y_sents))
        # print(type(sent_indexes))
        # print(type(total_y_sents))

        r = np.random.randint(0, len(total_y_sents))
        print("rand: ", r)
        print([i for i in sent_indexes])

        X_train = [self.sent2features(s, feature_select) for s in [sents[i] for i in sent_indexes]] + [self.sent2features(sents[r], feature_select)] +[self.sent2features(sents[-1], feature_select)] + [self.sent2features(sents[-2], feature_select)]
        y_train = [self.sent2labels(s) for s in y_sents] + [self.sent2labels(total_y_sents[r])] + [self.sent2labels(total_y_sents[-1])] + [self.sent2labels(total_y_sents[-2])]

        X_test = [self.sent2features(s, feature_select) for s in [sents[i] for i in sent_indexes]] + [self.sent2features(sents[r], feature_select)] + [self.sent2features(sents[-1], feature_select)] + [self.sent2features(sents[-2], feature_select)]
        y_test = [self.sent2labels(s) for s in y_sents] + [self.sent2labels(total_y_sents[r])] + [self.sent2labels(total_y_sents[-1])] + [self.sent2labels(total_y_sents[-2])]

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

    # def get_unmapped_data(self):
    #     data_df = pd.read_csv(DATA_INP + 'extracted_data.csv', encoding='utf-8', header=None)



def main():
    print("Streaming data pipeline started...")

    stream = Data_Stream()
    start = time.time()
    _ = stream.gen_pattern(keywords = None)
    print("time : ", time.time() - start)
    start = time.time()
    _, _, _ = stream.tag_data()
    print("time : ", time.time() - start)
    start = time.time()
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