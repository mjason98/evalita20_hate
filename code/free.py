from .pyfreeling import*
import sys, os
from .utils import *
from senticnet.senticnet import SenticNet
import numpy as np
import re
import pandas as pd
import code

class LangFreeLing:
  def __init__(self, lang='it', logs=False):
    super(LangFreeLing, self).__init__()
    
    if "FREELINGDIR" not in os.environ :
       if sys.platform == "win32" or sys.platform == "win64" : os.environ["FREELINGDIR"] = "C:\\Program Files"
       else : os.environ["FREELINGDIR"] = "/usr/local"

       if logs:
        print("FREELINGDIR environment variable not defined, trying ", os.environ["FREELINGDIR"], file=sys.stderr)

    if not os.path.exists(os.environ["FREELINGDIR"]+"/share/freeling") :
         print("Folder",os.environ["FREELINGDIR"]+"/share/freeling",
               "not found.\nPlease set FREELINGDIR environment variable to FreeLing installation directory",
               file=sys.stderr)
         sys.exit(1)

    DATA = os.environ["FREELINGDIR"]+"/share/freeling/"
    util_init_locale("default")

    LANG=lang
    self.op= maco_options(LANG)
    self.op.set_data_files( "", 
                           DATA + "common/punct.dat",
                           DATA + LANG + "/dicc.src",
                           DATA + LANG + "/afixos.dat",
                           "",
                           DATA + LANG + "/locucions.dat", 
                           DATA + LANG + "/np.dat",
                           "",#DATA + LANG + "/quantities.dat",
                           DATA + LANG + "/probabilitats.dat")

    # create analyzers
    self.tk=tokenizer(DATA+LANG+"/tokenizer.dat")
    self.sp=splitter(DATA+LANG+"/splitter.dat")
    #self.sid=self.sp.open_session();
    self.mf=maco(self.op);

    # activate mmorpho modules to be used in next call
    self.mf.set_active_options(False, True, True, True,  # select which among created 
                          True, True, False, True,  # submodules are to be used. 
                          True, True, True, True ); # default: all created submodules are used

    # create tagger
    self.tg=hmm_tagger(DATA+LANG+"/tagger.dat",True,2)

  def reducto(selr, text):
    mi = min(len(text),4)
    return text[:mi]

  def initPrep(self, text):
    lin= text
    l = self.tk.tokenize(lin)
    ls = self.sp.split(l)
    ls = self.mf.analyze(ls)
    ls = self.tg.analyze(ls)
    return ls

  def sentences(self, text, s_fun=None):
    sol = []
    ls = self.initPrep(text)
    ## output results
    for s in ls :
      ws = s.get_words()
      sent = []
      for w in ws :
        sent.append(w.get_form())
      sent = ' '.join(sent)
      if s_fun is not None:
        sent = s_fun(sent)
      sol.append(sent)
    return sol

  def tagizar(self, text):
    ls= self.initPrep(text)
    sol = []

    ## output results
    for s in ls :
       ws = s.get_words()
       for w in ws :
          sol.append(self.reducto(w.get_tag()))
    return ' '.join(sol)

  def lematizar(self, text):
    ls= self.initPrep(text)
    sol = []

    for s in ls :
       ws = s.get_words()
       for w in ws :
          sol.append(w.get_lemma())
    return ' '.join(sol)

class stopsIT:
  def __init__(self,path='data/stopwords-it.txt'):
    self.fo = None

    with open(path, 'r') as file:
      s = file.read().split()
      s = '|'.join(' ' + i + ' ' for i in s)
      self.fo = re.compile(s)
  def remove(self,text):
    text = self.fo.sub(' ', text)
    return text

FREE = LangFreeLing('it')
SN   = SenticNet('it')

def makeHurtDict(path='hurtlex_IT.tsv'):
  path = os.path.join(code.__path__[0], path)

  position  = ['ps', 'rci', 'pa', 'ddf', 'ddp', 'dmc', 'is', 'or', 'an', 'asm', 'asf', 'pr', 'om', 'qas', 'cds', 're', 'svp']
  position  = dict([(tag, i) for i, tag in enumerate(position)])
  
  sol = {}

  pos_l = len(position)
  with open(path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      lin = line.lower().split('\t')
      if lin[0] == 'id':
        continue
      vec = [0]*(pos_l+1)
      vec[ position[lin[2]] ] = 1
      if lin[3] == 'yes':
        vec[-1] = 1 
      word = FREE.lematizar(lin[4])
      sol.update({word:vec})
  return sol

hurt_dict = makeHurtDict()

def getHurtF(texto):
  sol, eps = 0, 1e-9
  text = texto.lower()

  l1, l2 = 0, 5
  for firt in hurt_dict:
    l1 = len(hurt_dict[firt])
    break
  sol += np.array([0]*(l1+l2), dtype=np.float32)

  for word in text.split():
    try:
      vec = hurt_dict[word]
    except:
      vec = [0]*l1
    try:
      cons = SN.concept(word)
      vec = vec + [ cons['polarity_value'], cons['sentics']['pleasantness'],
                    cons['sentics']['attention'], cons['sentics']['sensitivity'],
                    cons['sentics']['aptitude'] ]
    except:
      vec = vec + [0]*l2

    sol += np.array(vec, dtype=np.float32)
  div = np.array([ float(1. / (np.sqrt((sol[:-5]*sol[:-5]).sum()) + eps)) ]*l1 + [1.]*l2, dtype=np.float32)
  return sol * div

def clear_free():
  del SN
  del FREE
  del hurt_dict