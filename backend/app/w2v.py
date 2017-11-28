import tensorflow as tf
import numpy as np

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

def main():
    X, y, wordidx = build_data()
    print(X)

if __name__ == '__main__':
    main()
