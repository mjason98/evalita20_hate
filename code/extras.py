import pickle
import os
import pandas as pd 
import numpy as np 

from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA, TruncatedSVD

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from nltk.corpus import wordnet as wn

from .utils import StatusBar

GLOBAL_DICT = {}
SYN_DICT = {}
SYN_PATH_DICT = {}

def getWordNet_vector(text, seq_len=20):
    global GLOBAL_DICT

    text = text.split()
    if len(text) <= seq_len:
        text = text + [' ']*(seq_len - len(text))
    else:
        text = text[:seq_len]

    di = {'n': 0, 'a': 0, 's': 0, 'v': 0, 'r': 0, 'l': 0}
    for p in text:
        syns = wn.synsets(p)
        for sy in syns:
            di[sy.pos()] += 1
        di['l'] += len(syns)
    vec = np.array([di[a] for a in di], dtype=np.float32)
    return vec


def get_syn(word):
    global SYN_DICT

    tmp = SYN_DICT.get(word)
    if tmp is not None:
        return tmp
    tmp = wn.synsets(word)

    if len(tmp) > 0:
        up = {word: tmp}
        SYN_DICT.update(up)
        return tmp
    else:
        return None

def get_path_valor(syn1, syn2):
    clave = (syn1, syn2)
    tmp = SYN_PATH_DICT.get(clave)
    if tmp is not None:
        return tmp
    val = syn1.path_similarity(syn2)
    if val is None:
        val = 0.
    up = {clave: val}
    SYN_PATH_DICT.update(up)
    return val


def get_text_2Vec(text, seq_len=32):
    text = text.split()
    if len(text) <= seq_len:
        text = text + [' '] * (seq_len - len(text))
    else:
        text = text[:seq_len]

    val = []
    for p in text:
        v,c = 0, 0
        for p2 in text:
            s1 = get_syn(p)
            s2 = get_syn(p2)

            if s1 is None or s2 is None:
                continue

            for sins in s1:
                for sins2 in s2:
                    v += get_path_valor(sins, sins2)
        val.append(v)

    return np.array(val, dtype=np.float32)

def get_feature_vector(text, SEQ_LEN=10, eps=1e-9):
    v    = getWordNet_vector(text, SEQ_LEN)
    v2   = get_text_2Vec(text, SEQ_LEN)
    vec  = np.zeros(6 + SEQ_LEN)
    vec[:6] += v / (np.sqrt((v*v).sum()) + eps )
    vec[6:] += v2 / (np.sqrt((v2*v2).sum()) + eps )
    return vec

#------------------------------------------------------

def makeArrayVector_evalita2020(csv_path, dim=2):
    DATA, ID = [], []
    data = pd.read_csv(csv_path)

    bar = StatusBar(len(data), title='# Data')
    for i in range(len(data)):
        vec = data.iloc[i, 3]
        ide = int(data.iloc[i, 2])
        vec = [float(i) for i in vec.split()]

        DATA.append(vec)
        ID.append(int(ide))
        bar.update()
    del data 
    return np.array(DATA, dtype=np.float32), ID

def Proyecy2Data(np_data, ides=None):
    print ('# Training embedding model..')
    assert len(np_data.shape) == 2

    X_embb = TSNE(n_components=2).fit_transform(np_data)
    #X_embb = PCA(n_components=2, svd_solver='full').fit_transform(np_data)
    #X_embb = TruncatedSVD(n_components=2).fit_transform(np_data)
    print ('# Done!')

    if ides:
        D_1, D_2 = [], []
        bar = StatusBar(len(ides), title='# Color')
        for pos,i in enumerate(ides):
            if i == 0:
                D_1.append(X_embb[pos].reshape(1,-1))
            else:
                D_2.append(X_embb[pos].reshape(1,-1))
            bar.update()
        del X_embb

        D_1 = np.concatenate(D_1, axis=0)
        D_2 = np.concatenate(D_2, axis=0)
        
        plt.scatter(D_1[:,0], D_1[:,1])
        plt.scatter(D_2[:,0], D_2[:,1])
        plt.show()
    else:
        plt.scatter(X_embb[:,0], X_embb[:,1])
        plt.show()

def Proyecy3Data(np_data, ides=None):
    print ('# Training embedding model..')
    assert len(np_data.shape) == 2

    X_embb = TSNE(n_components=3).fit_transform(np_data)
    #X_embb = PCA(n_components=3, svd_solver='full').fit_transform(np_data)
    #X_embb = TruncatedSVD(n_components=3).fit_transform(np_data)
    print ('# Done!')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if ides:
        D_1, D_2 = [], []
        bar = StatusBar(len(ides), title='# Color')
        for pos,i in enumerate(ides):
            if i == 0:
                D_1.append(X_embb[pos].reshape(1,-1))
            else:
                D_2.append(X_embb[pos].reshape(1,-1))
            bar.update()
        del X_embb

        D_1 = np.concatenate(D_1, axis=0)
        D_2 = np.concatenate(D_2, axis=0)

        ax.scatter(D_1[:,0], D_1[:,1], D_1[:,2], marker='o')
        ax.scatter(D_2[:,0], D_2[:,1], D_2[:,2], marker='^')
        
        plt.show()
    else:
        ax.scatter(X_embb[:,0], X_embb[:,1], X_embb[:,2], marker='o')
        plt.show()