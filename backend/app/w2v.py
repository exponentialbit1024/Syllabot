import tensorflow as tf
import numpy as np
from dbOb import dbOb
import pickle
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import math
import collections
import random
import datetime as dt
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
from config import *
import re
from sklearn.neighbors import NearestNeighbors
from nltk import bigrams

def get_mysql_data():
    dobj = dbOb()
    tcursor = dobj.db.cursor()
    fetchQry = "SELECT data FROM syllabotData"
    tcursor.execute(fetchQry, [])
    allFetch = tcursor.fetchall()
    allSents = [x[0] for x in allFetch]
    print(allSents)
    return allSents

def tokenization_cleaning(allSents):
    tokens = [word_tokenize(report.lower()) for report in allSents]
    regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html
    punct_remove = []
    for sent in tokens:
        new_tokens = []
        for token in sent:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_tokens.append(new_token)

        punct_remove.append(new_tokens)

    stop_remove = []
    for sent in punct_remove:
        new_term_vector = []
        for word in sent:
            if not word in stopwords.words('english'):
                new_term_vector.append(word)
        stop_remove.append(new_term_vector)
    return punct_remove, stop_remove

def knn_run(vectors, subject):
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(vectors)
    distances, indices = nbrs.kneighbors([subject])
    return indices

def join_bigrams(stop_remove):
    target_bigram_list = [('cs', '252'), ('email', 'id')]
    replace_bigram_list = ['cs_252', 'email_id']
    new_stop_remove = []
    for sent in stop_remove:
        new_sent = []
        bgrm = list(bigrams(sent))
        for tbgrm in target_bigram_list:
            if tbgrm in bgrm:
                new_sent.append(replace_bigram_list[target_bigram_list.index(tbgrm)])
                for bw in tbgrm:
                    sent.pop(sent.index(bw))
        for word in sent:
            new_sent.append(word)
        new_stop_remove.append(new_sent)
    return new_stop_remove

def attach_important_stops(punct_remove, stop_remove, wv, wordidx):
    question_stops = set(["what", "how", "when", "where", "which"])
    new_final_tokens = []
    # print("stop_remove", stop_remove)
    stop_remove = join_bigrams(stop_remove)
    for sent in stop_remove:
        if len(sent) == 1:
            idx = stop_remove.index(sent)
            punct_map = set(punct_remove[idx])
            hit_stop = list(set(punct_map).intersection(question_stops))
            if len(hit_stop) > 0:
                indices = knn_run(wv, wv[wordidx.index(hit_stop[0])])
                for index in indices[0][1:]:
                    if wordidx[index] not in question_stops:
                        sent.insert(0, wordidx[index])
                        break
            else:
                new_final_tokens.append(sent)
        new_final_tokens.append(sent)
    return new_final_tokens

def cleaned_input(wv, widx):
    allSents = get_mysql_data()
    punct_remove, stop_remove = tokenization_cleaning(allSents)
    final_tokens = attach_important_stops(punct_remove, stop_remove, wv, widx)
    return final_tokens

def sort_mysql_data(final_tokens):
    f = open("./data/allData", "w")
    for list_token in final_tokens:
        f.write(' '.join(list_token))
        f.write("\n")
    f.close()

def build_data():
    pf = open("./data/pairings", "r")
    pairs = pf.read().split("\n")
    allwords = []
    for pair in pairs:
        pairword = pair.split(",")
        for pw in pairword:
            if pw not in allwords:
                allwords.append(pw)
    # print(allwords)
    pf.close()
    X = np.eye(len(allwords), dtype=int)
    pair2d = []
    for pair in pairs:
        pairsplit = pair.split(",")
        pair2d.append(pairsplit)

    y = np.zeros((len(allwords), len(allwords)), np.int)

    r = 0
    for x in X:
        idx = np.where(x == 1)[0][0]
        iw = allwords[idx]
        for p1d in pair2d:
            if iw in p1d:
                for ws in p1d:
                    y[r][allwords.index(ws)] = 1
        r += 1
    return X, y, allwords

def gen_w2v(X, y):
    x_train = np.asarray(X)
    y_train = np.asarray(y)
    vocab_size = len(X)
    x = tf.placeholder(tf.float32, shape=(None, vocab_size))
    y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

    EMBEDDING_DIM = 3
    W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
    b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
    hidden_representation = tf.add(tf.matmul(x,W1), b1)

    W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
    b2 = tf.Variable(tf.random_normal([vocab_size]))
    prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

    n_iters = 10000
    for _ in range(n_iters):
        sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
        print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

    vectors = sess.run(W1 + b1)
    return vectors

def plot_vectors(wvs, wordidx):
    pca = PCA(n_components=2)
    result = pca.fit_transform(wvs)
    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(wordidx):
    	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

def main():
    buildFlag = None
    mysqlPullFlag = False
    if len(sys.argv) > 1:
        buildFlag = sys.argv[1]
        if "--data-pull=true" in sys.argv:
            mysqlPullFlag = True
    X, y, wordidx = build_data()
    word_vec_file = "wv.pkl"
    wvs = None
    if buildFlag == '-n' or buildFlag is None:
        wvs = gen_w2v(X, y)
        with open(word_vec_file, "wb") as f:
            pickle.dump(wvs, f)
    elif buildFlag == "-o":
        with open(word_vec_file, "rb") as f:
            wvs = pickle.load(f)

    if wvs is None:
        print("error loading w2v model")
        sys.exit(-1)

    # plot_vectors(wvs, wordidx)
    if mysqlPullFlag:
        final_tokens = cleaned_input(wvs, wordidx)
        sort_mysql_data(final_tokens)

if __name__ == '__main__':
    main()
