import pickle
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import math
import collections
import random
import re
from nltk import bigrams
from sklearn.neighbors import NearestNeighbors
from nltk.tokenize import RegexpTokenizer

RESPONSE_CP = "CS 252 holds class in WALC 1018"
RESPONSE_CDT = "The class is every Monday, Wednesday and Friday from 11:30 AM - 12:30 PM, unless stated otherwise"
RESPONSE_EPM = "The midterm is in PHYS 112 and PHYS 114"
RESPONSE_EPF = "The final is in STEW 183 also known as Loeb Playhouse"
RESPONSE_EDTM = "The midterm is on October 16, 2017 from 8:00 PM - 10::00 PM"
RESPONSE_EDTF = "The final is on December 16, 2017 from 3:30 PM - 5:30 PM"
RESPONSE_COMBC = "CS 252 has classes is every Monday, Wednesday and Friday from 11:30 AM - 12:30 PM, in WALC 1018, unless stated otherwise"
RESPONSE_COMBEM = "The midterm is on October 16, 2017 from 8:00 PM - 10::00 PM un PHYS 112 and PHYS 114"
RESPONSE_COMBEF = "The final is on December 16, 2017 from 3:30 PM - 5:30 PM in STeW 183 (Loeb Playhouse)"
RESPONSE_GRADING = "The final grade will be about 50" + '%'  + " midterm and final exams, 40" + '%'  + " projects and homeworks, 10" + '%' + " attendance."
RESPONSE_ATTENDENCE = "The attendance is worth 10" + '%' + " of the grade."
RESPONSE_FINAL = "The final is worth 25" + '%' + " of the grade."
RESPONSE_MIDTERM = "The midterm is worth 25" + '%' + " of the grade."
RESPONSE_PROJECT = "The projects and homeworks are worth 40" + '%' + " of the grade."
RESPONSE_EMAIL = "The course email ID is grr@cs.purdue.edu"
RESPONSE_PROF = "You can contact the professor at grr@cs.purdue.edu"
RESPONSE_LATE = "10" + '%' + " of your grade for the project will be deducted for each late day, unless there is a hard deadline, which will result in a 0."
RESPONSE_MAN_ATT = "The attendence is not mandatory, but there will be points for attendance which account for 10" + '%' + " of your final grade."
REPONSE_CLASS_INFO = " Low-level programming; review of addresses, pointers, memory layout, and data representation; text, data, and bss segments; debugging and hex dumps; concurrent execution with threads and processes; address spaces; file names; descriptors and file pointers; inheritance; system calls and library functions; standard I/O and string libraries; simplified socket programming; building tools to help programmers; make and make files; shell scripts and quoting; unix tools including sed, echo, test, and find; scripting languages such as awk; version control; object and executable files (.o and a.out); symbol tables; pointers to functions; hierarchical directories; and DNS hierarchy; programming embedded systems."

class classifier:

    def __init__(self):
        self.model = load_model('./app/chatlstm.h5')
        self.vocab = np.load("./app/data/vocab.npy")
        self.vocab_map = self.create_vocab_dict(list(self.vocab))
        with open("./app/wv.pkl", "rb") as f:
            self.wv = pickle.load(f)

    def create_vocab_dict(self, vocab):
        vocab_map = {}
        for word in vocab:
            vocab_map[word] = vocab.index(word) + 1
        return vocab_map

    def knn_run(self, vectors, subject):
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(vectors)
        distances, indices = nbrs.kneighbors([subject])
        return indices

    def tranform_new_sample(self, sentence, wv, wordidx):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence.lower())
        stop_remove = []
        for token in tokens:
            if not token in stopwords.words('english'):
                stop_remove.append(token)

        target_bigram_list = [('cs', '252'), ('email', 'id'), ('final', 'exam'), ('midterm', 'exam')]
        replace_bigram_list = ['cs_252', 'email_id', 'final_exam', 'midterm_exam']
        bigram_transform = []
        bgrm = list(bigrams(stop_remove))
        for tbgrm in target_bigram_list:
            if tbgrm in bgrm:
                bigram_transform.append(replace_bigram_list[target_bigram_list.index(tbgrm)])
                for bw in tbgrm:
                    stop_remove.pop(stop_remove.index(bw))
        for word in stop_remove:
            bigram_transform.append(word)
        question_stops = set(["what", "how", "when", "where", "which"])

        if len(bigram_transform) == 1:
            hit_stop = list(set(tokens).intersection(question_stops))
            if len(hit_stop) > 0:
                for hs in hit_stop:
                    indices = self.knn_run(wv, wv[wordidx.index(hs)])
                    for index in indices[0][1:]:
                        if wordidx[index] not in question_stops:
                            bigram_transform.insert(0, wordidx[index])
                            break

        return bigram_transform

    def map_new_sequence(self, bgm, vocab_map):
        seq = []
        maxLen = 5
        if len(bgm) == 1 and bgm[0] not in vocab_map:
            return -1
        for word in bgm:
            try:
                seq.append(vocab_map[word])
            except:
                print(word, "not in vocab")
        while len(seq) < maxLen:
            seq.insert(0, 0)
        return seq

    def map_response(self, prediction, bigram):
        if prediction == 1:
            return RESPONSE_CP
        elif prediction == 2:
            return RESPONSE_CDT
        elif prediction == 3:
            #exam place
            return RESPONSE_EPF
        elif prediction == 4:
            #exam time and place
            return RESPONSE_EDTF
        elif prediction == 5:
            return RESPONSE_COMBC
        elif prediction == 6:
            #combo exam info
            return RESPONSE_COMBEM + " and " + RESPONSE_COMBEF
        elif prediction == 7:
            #grading stuff
            if "final" in bigram:
                return RESPONSE_FINAL
            elif "midterm" in bigram:
                return RESPONSE_MIDTERM
            return RESPONSE_GRADING
        elif prediction == 8:
            #misc
            if "attendance" in bigram:
                return RESPONSE_ATTENDENCE
            return RESPONSE_EMAIL
        elif prediction == 9:
            return RESPONSE_EPM
        elif prediction == 10:
            return RESPONSE_EDTM
        elif prediction == 11:
            return RESPONSE_COMBEM
        elif prediction == 12:
            return RESPONSE_EPF
        elif prediction == 13:
            return RESPONSE_EDTF
        elif prediction == 14:
            return RESPONSE_COMBEF
        elif prediction == 15:
            return REPONSE_CLASS_INFO

    def predict(self, inputSent):
        bigram_transform = self.tranform_new_sample(inputSent, self.wv, list(self.vocab))
        mappedInputSeq = self.map_new_sequence(bigram_transform, self.vocab_map)
        if mappedInputSeq != -1:
            pred_vec = self.model.predict(np.array(mappedInputSeq).reshape(1, 5))
            prediction = [np.argmax(vector) for vector in pred_vec]
            prediction_class = prediction[0]
            prediction_response = self.map_response(prediction_class, bigram_transform)
            return prediction_response
        return "Sorry I couldn\'t understand that"
