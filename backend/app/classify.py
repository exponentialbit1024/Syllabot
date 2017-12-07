import sys
import pickle
import numpy as np
import os
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

def save_sequence(sequence, cluster, index):
    if cluster == 1:
        np.save("./data/clusters/cp/" + str(index) + ".npy", sequence)
    elif cluster == 2:
        np.save("./data/clusters/cdt/" + str(index) + ".npy", sequence)
    elif cluster == 3:
        np.save("./data/clusters/ep/" + str(index) + ".npy", sequence)
    elif cluster == 4:
        np.save("./data/clusters/edt/" + str(index) + ".npy", sequence)
    elif cluster == 5:
        np.save("./data/clusters/combc/" + str(index) + ".npy", sequence)
    elif cluster == 6:
        np.save("./data/clusters/combe/" + str(index) + ".npy", sequence)
    elif cluster == 7:
        np.save("./data/clusters/grading/" + str(index) + ".npy", sequence)
    elif cluster == 8:
        np.save("./data/clusters/misc/" + str(index) + ".npy", sequence)

def load_sequence(cluster, index):
    if cluster == 1:
        seq = np.load("./data/clusters/cp/" + str(index) + ".npy")
        return seq
    elif cluster == 2:
        seq = np.load("./data/clusters/cdt/" + str(index) + ".npy")
        return seq
    elif cluster == 3:
        seq = np.load("./data/clusters/ep/" + str(index) + ".npy")
        return seq
    elif cluster == 4:
        seq = np.load("./data/clusters/edt/" + str(index) + ".npy")
        return seq
    elif cluster == 5:
        seq = np.load("./data/clusters/combc/" + str(index) + ".npy")
        return seq
    elif cluster == 6:
        seq = np.load("./data/clusters/combe/" + str(index) + ".npy")
        return seq
    elif cluster == 7:
        seq = np.load("./data/clusters/grading/" + str(index) + ".npy")
        return seq
    elif cluster == 8:
        seq = np.load("./data/clusters/misc/" + str(index) + ".npy")
        return seq

def generate_wvs_sequence(wvs, wordidx, sentences, maxLen, cluster):
    padding_vec = np.array([0, 0, 0])
    for sent in sentences:
        sequence = []
        words = sent.split(" ")
        for word in words:
            try:
                sequence.append(wvs[list(wordidx).index(word)])
            except ValueError:
                print(word, "not in vocab")
        while len(sequence) != maxLen:
            sequence.insert(0, padding_vec)
        save_sequence(sequence, cluster, sentences.index(sent))

def load_cluster_sequences(cluster):
    sequence_list = []
    if cluster == 1:
        for filename in os.listdir("./data/clusters/cp/"):
            if filename.endswith(".npy"):
                sequence_list.append(np.load("./data/clusters/cp/" + filename))
        return sequence_list
    elif cluster == 2:
        for filename in os.listdir("./data/clusters/cdt/"):
            if filename.endswith(".npy"):
                sequence_list.append(np.load("./data/clusters/cdt/" + filename))
        return sequence_list
    elif cluster == 3:
        for filename in os.listdir("./data/clusters/ep/"):
            if filename.endswith(".npy"):
                sequence_list.append(np.load("./data/clusters/ep/" + filename))
        return sequence_list
    elif cluster == 4:
        for filename in os.listdir("./data/clusters/edt/"):
            if filename.endswith(".npy"):
                sequence_list.append(np.load("./data/clusters/edt/" + filename))
        return sequence_list
    elif cluster == 5:
        for filename in os.listdir("./data/clusters/combc/"):
            if filename.endswith(".npy"):
                sequence_list.append(np.load("./data/clusters/combc/" + filename))
        return sequence_list
    elif cluster == 6:
        for filename in os.listdir("./data/clusters/combe/"):
            if filename.endswith(".npy"):
                sequence_list.append(np.load("./data/clusters/combe/" + filename))
        return sequence_list
    elif cluster == 7:
        for filename in os.listdir("./data/clusters/grading/"):
            if filename.endswith(".npy"):
                sequence_list.append(np.load("./data/clusters/grading/" + filename))
        return sequence_list
    elif cluster == 8:
        for filename in os.listdir("./data/clusters/misc/"):
            if filename.endswith(".npy"):
                sequence_list.append(np.load("./data/clusters/misc/" + filename))
        return sequence_list

def load_cluster_sents(clustername):
    cpf = open("./data/" + clustername)
    cp_sents = cpf.read().split("\n")
    cpf.close()
    cp_sents.pop(len(cp_sents) - 1)
    return cp_sents

def get_max_sequence_length():
    f = open("./data/allData")
    sents = f.read().split("\n")
    f.close()
    sents.pop(len(sents) - 1)
    maxLen = len(sents[0].split(" "))
    for sent in sents:
        if len(sent.split(" ")) > maxLen:
            maxLen = len(sent.split(" "))
    return maxLen

def create_vocab_dict(vocab):
    vocab_map = {}
    for word in vocab:
        vocab_map[word] = vocab.index(word) + 1
    return vocab_map

def create_lstm_sequence(vocab_map, sentences, cluster, maxLen=5):
    for sent in sentences:
        sequence = []
        for word in sent.split(" "):
            try:
                sequence.append(vocab_map[word])
            except:
                print(word, "not in vocab")
        while len(sequence) != maxLen:
            sequence.insert(0, 0)
        save_sequence(sequence, cluster, sentences.index(sent))

def build_model():
    embedding_vecor_length = 3
    model = Sequential()
    model.add(Embedding(76, embedding_vecor_length, input_length=5))
    model.add(LSTM(100))
    model.add(Dense(9, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def new_bigram_transform(token_list):
    target_bigram_list = [('cs', '252'), ('email', 'id'), ('final', 'exam'), ('midterm', 'exam')]
    replace_bigram_list = ['cs_252', 'email_id', 'final_exam', 'midterm_exam']
    new_sent = []
    bgrm = list(bigrams(token_list))
    for tbgrm in target_bigram_list:
        if tbgrm in bgrm:
            new_sent.append(replace_bigram_list[target_bigram_list.index(tbgrm)])
            for bw in tbgrm:
                token_list.pop(token_list.index(bw))
    for word in token_list:
        new_sent.append(token_list)
    return new_sent

def knn_run(vectors, subject):
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(vectors)
    distances, indices = nbrs.kneighbors([subject])
    return indices

def tranform_new_sample(sentence, wv, wordidx):
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
                indices = knn_run(wv, wv[wordidx.index(hs)])
                for index in indices[0][1:]:
                    if wordidx[index] not in question_stops:
                        bigram_transform.insert(0, wordidx[index])
                        break

    return bigram_transform

def map_new_sequence(bgm, vocab_map):
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

def main():
    word_vec_file = "wv.pkl"
    with open(word_vec_file, "rb") as f:
        wvs = pickle.load(f)

    maxLen = get_max_sequence_length()
    vocab = np.load("./data/vocab.npy")

    vocab_map = create_vocab_dict(list(vocab))

    if "--new-vecs=true" in sys.argv:
        cp_sents = load_cluster_sents("class_location")
        create_lstm_sequence(vocab_map, cp_sents, 1)

        cdt_sents = load_cluster_sents("class_time")
        create_lstm_sequence(vocab_map, cdt_sents, 2)

        ep_sents = load_cluster_sents("exam_location")
        create_lstm_sequence(vocab_map, ep_sents, 3)

        edt_sents = load_cluster_sents("exam_time")
        create_lstm_sequence(vocab_map, edt_sents, 4)

        cpdt_sents = load_cluster_sents("class_info")
        create_lstm_sequence(vocab_map, cpdt_sents, 5)

        epdt_sents = load_cluster_sents("exam_info")
        create_lstm_sequence(vocab_map, epdt_sents, 6)

        grading_sents = load_cluster_sents("grading")
        create_lstm_sequence(vocab_map, grading_sents, 7)

        misc_sents = load_cluster_sents("misc")
        create_lstm_sequence(vocab_map, misc_sents, 8)

    # cp_sents = load_cluster_sents("class_location")
    # cdt_sents = load_cluster_sents("class_time")
    # ep_sents = load_cluster_sents("exam_location")
    # edt_sents = load_cluster_sents("exam_time")
    # cpdt_sents = load_cluster_sents("class_info")
    # epdt_sents = load_cluster_sents("exam_info")
    # grading_sents = load_cluster_sents("grading")
    # misc_sents = load_cluster_sents("misc")
    # allSents = cp_sents + cdt_sents + ep_sents + edt_sents + cpdt_sents + epdt_sents + grading_sents + misc_sents

    c1 = load_cluster_sequences(1)
    c2 = load_cluster_sequences(2)
    c3 = load_cluster_sequences(3)
    c4 = load_cluster_sequences(4)
    c5 = load_cluster_sequences(5)
    c6 = load_cluster_sequences(6)
    c7 = load_cluster_sequences(7)
    c8 = load_cluster_sequences(8)

    X = c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8
    y = np.hstack((np.ones(len(c1)), 2 * np.ones(len(c2)), 3 * np.ones(len(c3)), 4 * np.ones(len(c4))))
    y = np.hstack((y, 5 * np.ones(len(c5)), 6 * np.ones(len(c6)), 7 * np.ones(len(c7)), 8 * np.ones(len(c8))))

    if "-n" in sys.argv:
        model = build_model()
        model.fit(np.array(X), to_categorical(y), epochs=300, batch_size=56)
        model.save("chatlstm.h5")
    elif "-o" in sys.argv:
        model = load_model('chatlstm.h5')

    # pred_vec = model.predict(np.array(X[0]).reshape(1, 5))
    # print([np.argmax(vector) for vector in pred_vec])
    print(len(vocab))
    while(True):
        inputSent = input("Enter test string: ")
        bigram_transform = tranform_new_sample(inputSent, wvs, list(vocab))
        mappedInputSeq = map_new_sequence(bigram_transform, vocab_map)
        if mappedInputSeq != -1:
            pred_vec = model.predict(np.array(mappedInputSeq).reshape(1, 5))
            print([np.argmax(vector) for vector in pred_vec])
        else:
            print("I\'m not sure what you want")

if __name__ == '__main__':
    main()
