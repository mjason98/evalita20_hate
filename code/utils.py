import math
import time
import numpy as np
import matplotlib.pyplot as plt
import random

import collections # reducer
import os # reducer
import re # reducer
import pickle

def colorizar(text):
    return '\033[91m' + text + '\033[0m'
def headerizar(text):
    return '\033[1m' + text + '\033[0m'

def getMyDict():
    return {'<emogy>':1, '<hashtag>':2, '<url>':3, '<risa>':4, '<signo>':5,
            '<ask>':6, '<phoria>':7, '<diag>':8, '<number>':9, '<date>':10,
            '<sent>':11, '<user>':12, '<frase>':13 }

def generate_dictionary_from_embedding(filename, dictionary, ret=True, logs=True, norm=False, message_logs=''):
    if logs:
        print ('# Loading:', colorizar(os.path.basename(filename)), message_logs)
    x = []
    band, l = False, 0

    mean, var, T = 0, 0, 0
    with open(filename, 'r', encoding='utf-8') as file:
        for ide, line in enumerate(file):
            li = line.split()

            if len(li) <= 2:
                print('#WARNING::', line, 'interpreted as head')
                continue
                
            if not band:
                x.append([0 for _ in range(len(li)-1)])
                my_d = getMyDict()
                l = len(my_d)
                for val in my_d:
                    x.append([random.random() for _ in range(len(li)-1)])
                    dictionary.update({val.lower(): my_d[val] })
                band = True

            a = [float(i) for i in li[1:]]
            x.append(a)
            
            mean += np.array(a, dtype=np.float32)
            var  += np.array(a, dtype=np.float32)**2
            T += 1

            dictionary.update({li[0].lower(): ide + l + 1})
    var  /= float(T)
    mean /= float(T)
    var -= mean ** 2
    var  = np.sqrt(var)
    mean = mean.reshape(1,mean.shape[0])
    var = var.reshape(1,var.shape[0])
    
    if ret:
        sol = np.array(x, np.float32)
        if norm:
            sol = (sol - mean) / var
        return sol

class StatusBar:
    def __init__(self, fin_, title=""):
        self.title = title
        self.ini = 0
        self.fin = fin_
        self.car = 's'
        self.ti = time.time()

    def update(self, loss=[], metrics=[]):
        self.ini = self.ini + 1
        metrics = " - ".join(['{}: {:.6f} '.format(n, m)
                              for n, m in loss + metrics])
        ctn = int(float(self.ini) / float(self.fin) * 20.0)
        bars = '=' * ctn + '.' * (20 - ctn)
        lo1, lo2 = int(math.log10(self.fin + 0.1) + 1), int(math.log10(self.ini + 0.1) + 1)
        spas = ' ' * (lo1 - lo2)
        ta = time.time() - self.ti
        if ta > 60:
            self.car = 'm'
            if ta / 60.0 > 60.0:
                self.car = 'h'
        if self.car == 'm':
            ta /= 60.0
        elif self.car == 'h':
            ta /= 360.0
        lo1 = 2 - int(math.log10(int(ta) + 0.1))
        spa2 = ' ' * lo1
        end = "" if self.ini < self.fin else "\n"
        print("\r{} {}{}/{} [{}] {}{:.1f}{} ".format(self.title, spas, self.ini, self.fin, bars, spa2, ta,
                                                       self.car) + metrics, end=end)  # + metrics, end=end)

class TorchBoard(object):
    def __init__(self):
        self.dict = {}
        self.labels = ['train', 'test']
        self.future_updt = True
        self.best_funct = None
        self.setFunct( max )
        self.best   = [None, None]
        self.best_p = [0, 0]

    def setFunct(self, fun):
        self.best_funct = fun

    def update(self, label, value, getBest=False):
        if self.future_updt == False:
            return
        if label not in self.labels:
            print ('WARNING: the label {} its not in {}, the board will not be updated.'.format(
                label, self.labels))
            self.future_updt = False
            return
        pk = 1
        if label == 'train':
            pk = 0

        if self.dict.get(label) == None:
            self.dict.update({label:[value]})
        else:
            self.dict[label].append(value)

        yo = False
        if self.best[pk] is None:
            yo = True
            self.best[pk] = value
        else:
            self.best[pk] = self.best_funct(self.best[pk], value)
            yo = self.best[pk] == value
        if yo:
            self.best_p[pk] = len(self.dict[label]) - 1
        if getBest:
            return yo

    def show(self, saveroute, plot_smood=False):
        fig , axes = plt.subplots()
        for i,l in enumerate(self.dict):
            y = self.dict[l]
            if len(y) <= 1:
                continue
            lab = str(self.best[i])
            if len(lab) > 7:
                lab = lab[:7]
            axes.plot(range(len(y)), y, label=l + ' ' + lab)
            axes.scatter([self.best_p[i]], [self.best[i]])

            if plot_smood:
                w = 3
                y_hat = [ np.array(y[max(i-w,0):min(len(y),i+w)]).mean() for i in range(len(y))]
                axes.plot(range(len(y)), y_hat, ls='--', color='gray')

        fig.legend()
        fig.savefig(saveroute)
        del axes
        del fig

# -------------------------------------------------
def reduced(oldFile, newFile, vocab):
    print ('# Turning', colorizar(oldFile), 'into', colorizar(newFile))
    
    file = open (newFile, 'w')
    with open(oldFile, 'r', encoding='utf-8') as oldf:
        for line in oldf.readlines():
            l = line.split()
            if len(l) <= 2:
                continue
            word = l[0].lower()
            if vocab.get(word, 0) != 0:
                file.write(line)
                vocab.pop(word)
    file.close()
    print('# Done!')

def makeVocabFromData(filepath):
    c = None
    with open(filepath, 'r', encoding='utf-8') as f:
        line = f.read().replace('\n', ' ')
        c = collections.Counter(line.split())

    return dict([(i, 5) for i in sorted(c, reverse=True)])