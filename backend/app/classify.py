import sys
import pickle
import numpy as np

def generate_wvs_sequence(wvs, words):
    return False

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

def main():
    word_vec_file = "wv.pkl"
    with open(word_vec_file, "rb") as f:
        wvs = pickle.load(f)

    maxLen = get_max_sequence_length()
    vocab = np.load("./data/vocab.npy")
    cp_sents = load_cluster_sents("class_location")
    cdt_sents = load_cluster_sents("class_time")
    ep_sents = load_cluster_sents("exam_location")
    edt_sents = load_cluster_sents("exam_time")

    cpdt_sents = load_cluster_sents("class_info")
    epdt_sents = load_cluster_sents("exam_info")

    misc_sents = load_cluster_sents("misc")
    grading_sents = load_cluster_sents("grading")

if __name__ == '__main__':
    main()
