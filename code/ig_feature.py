import pickle
import os
import pandas as pd 
import numpy as np 

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .preprocesing import IntSeq2
from .utils import StatusBar, colorizar

from sklearn.feature_extraction.text import CountVectorizer

BERT_PATH = '/DATA/Mainstorage/Prog/NLP/dbmdz/bert-base-italian-uncased'
bert_model = None
bert_tk = None

def calculate_ig_part(vec1, vec2, T, not_term=False):
    if not_term == False:
        vec1 = (vec1 == 0) * 1e-6 + vec1
        log_ = (vec1 * T) / (vec1.sum() * (vec1 + vec2))
        log_ = np.log2(log_)
        term = vec1 / T
        return term * log_
    else:
        log_ = ((T - vec1) * T) / ((vec1.sum() + vec2.sum() - vec1 - vec2) * (vec1.sum()))
        log  = np.log2(log_)
        term = 1.0 - vec1 / T
        return term * log_

def calculate_IG(data_raw, ctn=100, save_path=None):
    '''
        data_raw most be a pandas.Series and most have headers name: 'id', 'txt' and 'y'
    '''
    
    data, l = [], []
    bar = StatusBar(len(data_raw), title='# Data')
    for i in range(len(data_raw)):
        txt, y = data_raw.loc[i, 'txt'] , int(data_raw.loc[i,'y'])
        
        txt = IntSeq2(txt, None, None, only_sent_prep=True)
        data.append(txt)
        l.append(y)
        bar.update()

    print ('# Feature Selection')
    vectorizer = CountVectorizer(token_pattern=r'[^ ]+')
    data = vectorizer.fit_transform(data)
    data = data.toarray()
    l    = np.array(l)

    vec_0 = (l.reshape(-1,1) * data).sum(axis=0)
    vec_1 = ((1.0 - l).reshape(-1,1) * data).sum(axis=0)

    del data
    del l

    T    = vec_0.sum() + vec_1.sum()
    t_c  = calculate_ig_part(vec_0, vec_1, T) + calculate_ig_part(vec_1, vec_0, T)
    nt_c = calculate_ig_part(vec_0, vec_1, T, not_term=True) + calculate_ig_part(vec_1, vec_0, T, not_term=True)
    ig = t_c + nt_c

    d = [ (i, j) for j, i in zip(vectorizer.get_feature_names(), [a for a in ig])]
    d.sort()
    d = [i for j, i in d]

    del T
    del ig 
    del nt_c 
    del t_c 
    
    if ctn > len(d):
        print('# The amount of feature', ctn, 'is to big, the new sise is 50')
        ctn = 100
    d = d[-ctn:]

    if save_path is not None:
        file = open(save_path, 'wb')
        pickle.dump(d, file)
        file.close()

    return d

def selector_list_featurizer(flist, data_raw, save_file_path):
    '''
        data_raw most be a pandas.Series and most have headers name: 'id', 'txt' and 'y'
    '''
    fdic = dict([(n, i) for i, n in enumerate(flist)])

    F = []
    bar = StatusBar(len(data_raw), title='# Making ' + colorizar(os.path.basename(save_file_path)))
    for i in range(len(data_raw)):
        txt = data_raw.loc[i, 'txt']
        txt = IntSeq2(txt, None, None, only_sent_prep=True).split()
        feat = np.zeros(len(flist))
        for word in txt:
            p = fdic.get(word)
            if p is not None:
                feat[p] += 1.0
        feat = feat / (np.sqrt(feat*feat) + 1e-9)
        F.append(' '.join([str(v) for v in feat]))
        bar.update()

    F = pd.Series(F)
    data_id  = data_raw.drop(['txt', 'y'], axis=1)
    data_new = pd.concat([data_id, F], axis=1)
    data_new.to_csv(save_file_path, sep='\t',header=('id', 'feature'), index=None)

    del data_new
    del data_id
    del F

    return save_file_path

def recarge_bert():
    global bert_model
    global bert_tk
    bert_tk = AutoTokenizer.from_pretrained(BERT_PATH)
    bert_model = AutoModel.from_pretrained(BERT_PATH)

def delete_bert():
    global bert_model
    global bert_tk

    del bert_model
    del bert_tk

    bert_model = None
    bert_tk = None

def create_bert_features(data_raw, save_file_path, max_length=20):
    '''
        data_raw: pandas Series with header (id, txt, y)
        file_path_save: string, where to save a csv fetures
    '''

    assert (bert_model != None)
    assert (bert_tk != None)

    T = []
    bar = StatusBar(len(data_raw), title='# Bert-king ' + colorizar(os.path.basename(save_file_path)))
    for i in range(len(data_raw)):
        text = data_raw.loc[i, 'txt']
        with torch.no_grad():
            inputs = bert_tk(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
            out = bert_model(**inputs, output_hidden_states=True)

            # Add and Norm to the 2nd hidden layer
            feat = F.normalize(out.hidden_states[2].sum(dim=1), dim=-1).reshape(-1).numpy()
            T.append(' '.join([str(v) for v in feat]))
        bar.update()
    
    datid  = data_raw.drop(['txt', 'y'], axis=1)
    data_new = pd.concat([datid, pd.Series(T)], axis=1)
    data_new.to_csv(save_file_path, sep='\t',header=('id', 'feature'), index=None)