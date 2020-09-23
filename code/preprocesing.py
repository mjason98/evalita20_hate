from string import punctuation
import re
import os
import collections
import pandas as pd 
import random
import pickle

import code
from .utils  import *
from .free   import stopsIT, FREE, getHurtF
from .extras import get_feature_vector as getWordVec

class TokensMixed(object):
	"""docstring for TokensMixed"""
	def __init__(self, filepath, pre_vocab=None, logs=False, filtero=['adj', 'adv', 'nom', 'verb', 'int', 'tanc', 'vaux', 'noun', 'other']):
		super(TokensMixed, self).__init__()
		self.filter = filtero
		self.tree = self.armarArbol(filepath, logs, pre_vocab)


	def armarArbol(self, filepath, logs, pre_vocab):
		c = None
		if pre_vocab is not None:
			if logs:
				print('- Pre vocabulary passed, filepath descarted!')
			c = [p for p in pre_vocab]
		else:
			if logs:
				print ('- Starting to read', filepath)
			for file in os.listdir(filepath):
				if logs:
					print ('--', file)
				if len(re.findall('|'.join(self.filter), file)) < 1:
					if logs:
						print('--- not added')
					continue

				with open(os.path.join(filepath, file), 'r') as f:
					line = f.read().split()
					if c is None:
						c = collections.Counter(line)
					else:
						c += collections.Counter(line)

		texto = '|'.join(sorted(c, reverse=True))
		return re.compile(texto)

	def separar(self, text, lista=False, min_leng=1):
		text = text.lower()
		text = self.tree.findall(text)
		to = []
		for t in text:
			if len(t) >= min_leng:
				to.append(t)
		text = to 
		if lista == False:
			text = ' '.join(text)
		return text

def loadVocabPickle():
	vpath = os.path.join(code.__path__[0], 'default.voc')
	f = open(vpath, 'rb')
	voc = pickle.load(f)
	f.close()
	return voc

MY_DICT  = getMyDict()
STOPWORD = stopsIT()
TREE     = TokensMixed('', pre_vocab=loadVocabPickle())
gloval_punct = ''

def clear_preprocessing():
	del MY_DICT
	del STOPWORD
	del TREE
	del gloval_punct

def setTree_fromvocab(vocab):
	''' crea el arbol de palabras para hastag
	separator de un vovabulario '''
	global TREE
	TREE = TokensMixed('', pre_vocab=vocab)

def setTree_fromfile(folder_path):
	''' crea el arbol de palabras para hastag
	separator de una carpeta '''
	global TREE
	TREE = TokensMixed(folder_path)

def re_preprocessor(text):
    text = re.sub('URL', ' http:///www.mi.com ', text)

    text = re.sub('<[^>]*>', '', text).lower()
    text = re.sub("(?::|;|=)(?:-)?(?:\)|\(|D|P)", " <emogy> ", text)
    text = re.sub("[?¿]+", " <ask> ", text)
    text = re.sub("[!¡]+", " <phoria> ", text)
    text = re.sub("[—-]+", " <diag> ", text)
    text = re.sub('#[^ #]*', ' <hashtag> ', text)
    text = re.sub('@[^ @]*', ' <user> ', text)
    text = re.sub('http[s]?:[^ ]+', ' <url> ', text)
    text = re.sub('[hj]+[a]+([hj]+[a]+)+', ' <risa> ', text)
    text = re.sub("[0-9]+/[0-9]+/[0-9]+", " <date> ", text)
    text = re.sub("[0-9]+", " <number> ", text)
    text = re.sub("[,;:]", ' <signo> ', text)
    text = re.sub("[\'\"]", ' <frase> ', text)
    text = re.sub("[^\w<>]+", ' ', text)

    return text

def delRepet(text, max=2):
    global gloval_punct
    
    sol, i ='', 0
    for i in range(len(text)):
    	if i > 0 and text[i] == text[i-1] and text[i] in punctuation:
    		continue
    	if i > 1 and text[i] == text[i-1] and text[i] == text[i-2]:
    		continue
    	sol += text[i]

    if len(gloval_punct) < 1:
    	for c in punctuation:
        	if c not in "<>":
        		gloval_punct += c
    return sol

def hashtagWordSep(text):
	assert TREE != None
	text = text.split()
	solution = []

	for word in text:
		if word[0] == '#':
			solution.append('#')
			solution += TREE.separar(word, lista=True)
		else:
			solution.append(word)
	return ' '.join(solution)

def todo(text):
	text = FREE.lematizar(text)
	text = re.sub('_', ' ', text)
	text = re.sub('# ', '#', text)
	text = re.sub('@ ', '@', text)
	text = hashtagWordSep(text)
	text = STOPWORD.remove(text)
	text = re_preprocessor(text)
	return text

def clipSeq(seq, max_size):
	if len(seq) > max_size:
			return seq[:max_size]
	else:
		return seq + [0 for _ in range(max_size - len(seq))]

def sent_prep(text, function=todo, ret_list=False, counter=None):
	text = delRepet(text)
	sents = FREE.sentences(text, function)
		
	if counter is not None:
		counter[0] = max(counter[0], len(sents))
	if not ret_list:
		text  = ' <sent> '.join(sents)
		return text
	else:
		return sents

def IntSeq2(text, dictionary, max_size=50, mxs=None, feture=False):
	text = sent_prep(text)
	if feture:
		feat = list(getHurtF(text)) + list(getWordVec(text, max_size))
	text = text.split()

	if mxs is not None:
		mxs[0] += len(text)
		mxs[1] += len(text)**2

	S1  = [dictionary.get(tx, 0) for tx in text]
	S1 = clipSeq(S1, max_size)

	if feture:
		return S1, feat
	else:
		return S1

def sentBuild(sen):
	sen = re.sub('# ', '#', sen)
	sen = re.sub('@ ', '@', sen)
	sen = re.sub('_', ' ', sen)
	return sen

def makePrepData(filepath="data/name.tsv", vocab=None, seq_len=50, name_only=False, separete_sents=True):
	""" labels most be [id, txt, y] """
	new_name = filepath[:-4] + '_prep.csv'
	if name_only and os.path.isfile(filepath):
		return new_name

	data = pd.read_csv(filepath, sep='\t', encoding='utf-8')
	DATA = []
	bar = StatusBar(len(data), title='# Prep ' + headerizar(os.path.basename(filepath)))
	denom, stats = 0, [0,0]
	
	for i in range(len(data)):
		ide, text, hs = data.loc[i, 'id'], str(data.loc[i, 'txt']), data.loc[i, 'y']
		
		# Convert word sentence to dictionary index sentence

		if separete_sents:
			# Augment the data by separating the tweets into sentences
			
			sents = FREE.sentences(text)
			denom += len(sents)
			for sen in sents:
				sen = sentBuild(sen)
				
				sen, feat = IntSeq2(sen, vocab, max_size=seq_len, feture=True, mxs=stats)
				sen  = ' '.join([str(s) for s in sen])
				feat = ' '.join([str(s) for s in feat])

				DATA.append( (ide, sen, hs, feat) )
		else:
			# The size of the data remaing the same
			denom += 1
			sen, feat = IntSeq2(sen, vocab, max_size=seq_len, feture=True, mxs=stats)
			sen  = ' '.join([str(s) for s in sen])
			feat = ' '.join([str(s) for s in feat])
			DATA.append( (ide, sen, hs, feat) )

		bar.update()
		
	stats[0] /= denom
	stats[1] /= denom
	stats[1] -= stats[0] **2
	
	# Save the integer sequence in a new file
	A, B = pd.Series([i for i, _, _, _ in DATA]), pd.Series([i for _, i, _, _ in DATA])
	C, D = pd.Series([i for _, _, i, _ in DATA]), pd.Series([i for _, _, _, i in DATA])
	result = pd.concat([A,B,C,D], axis=1)
	result.to_csv(new_name, header=('id', 'txt', 'y', 'feat'), index=False)
	del A
	del B
	del C
	del result
	del DATA

	print('# Mean:', stats[0], 'Std:', stats[1] ** 0.5)
	return new_name

def EVALITA2018_HATE(filepath, task=None, validation=(0,1), name_only=False, names=['data/train.tsv', 'data/dev.tsv']):
	''' Task: string 'hs' or 'stereotype' '''
	if validation[1] > 1 and (validation[0] < 0 or validation[0] >= validation[1]):
		print ("# WARNING:: validation parameter", validation, 'invalid, seted to (0,{})'.format(validation[1]))
		validation = (0, validation[1])

	print ('# Preprocessing', colorizar(os.path.basename(filepath)))
	newR1,newR2  = names[0], names[1]

	if name_only:
		if validation[1] > 1:
			return newR1,newR2
		else:
			return newR1

	data = pd.read_csv(filepath, sep='\t', header=None, names=['id', 'text', 'label'])
	bar = StatusBar(len(data), title='# Data')
	
	ini, fin = -2,-1
	if validation[1] > 1:
		dval = len(data) / validation[1]
		ini  = validation[0]*dval
		fin  = (validation[0]+1)*dval

	DATA, VAL_DATA = [], []

	# Making a random iterator to generate the two portions of data
	ITER = None
	iter_path = os.path.basename(filepath)
	iter_path = re.sub('\.[.]*', '', iter_path)
	iter_path = os.path.join(code.__path__[0], 'iter_' + iter_path)
	if os.path.isfile(iter_path):
		file = open(iter_path, 'rb')
		ITER = pickle.load(file)
		file.close()
	else:
		ITER = [ i for i in range(len(data))]
		random.shuffle(ITER)
		file = open(iter_path, 'wb')
		pickle.dump(ITER, file)
		file.close()
	# ------------------------------------------------------------

	for j,i in enumerate(ITER):
		ide, text, hs = data.loc[i, 'id'], data.loc[i, 'text'], data.loc[i, 'label']
		if j >= ini and j < fin:
			VAL_DATA.append((ide, text, hs))
		else:
			DATA.append((ide, text, hs))
		bar.update()
	del data

	I = [i for i,_, _ in DATA]
	X = [x for _,x, _ in DATA]
	Y = [y for _,_, y in DATA]
	Ci, Cx, Cy = pd.Series(I), pd.Series(X), pd.Series(Y)
	del X
	del I
	del Y
	result = pd.concat([Ci, Cx, Cy], axis=1)
	result.to_csv(newR1, sep='\t', header=('id', 'txt', 'y'), index=False)
	del Cx
	del Ci
	del Cy
	del result

	if len(VAL_DATA) > 0:
		I = [i for i, _, _ in VAL_DATA]
		X = [x for _, x, _ in VAL_DATA]
		Y = [y for _, _, y in VAL_DATA]
		Ci, Cx, Cy = pd.Series(I), pd.Series(X), pd.Series(Y)
		del X
		del Y
		del I
		result = pd.concat([Ci, Cx, Cy], axis=1)
		result.to_csv(newR2, sep='\t', header=('id', 'txt', 'y'), index=False)
		del Cx
		del Cy
		del Ci
		del result

		return newR1, newR2
	else:
		return newR1

def EVALITA2020_HATE(filepath, task=None, validation=(0,1), name_only=False, names=['data/train.tsv', 'data/dev.tsv'], tsk='hs'):
	''' Task: string 'hs' or 'stereotype' '''
	if validation[1] > 1 and (validation[0] < 0 or validation[0] >= validation[1]):
		print ("# WARNING:: validation parameter", validation, 'invalid, seted to (0,{})'.format(validation[1]))
		validation = (0, validation[1])

	print ('# Preprocessing', colorizar(os.path.basename(filepath)))
	newR1,newR2  = names[0], names[1]

	if name_only:
		if validation[1] > 1:
			return newR1,newR2
		else:
			return newR1

	data = pd.read_csv(filepath, sep='\t')
	bar = StatusBar(len(data), title='# Data')
	
	ini, fin = -2,-1
	if validation[1] > 1:
		dval = len(data) / validation[1]
		ini  = validation[0]*dval
		fin  = (validation[0]+1)*dval

	DATA, VAL_DATA = [], []

	# Making a random iterator to generate the two portions of data
	ITER = None
	iter_path = os.path.basename(filepath)
	iter_path = re.sub('\.[.]*', '', iter_path)
	iter_path = os.path.join(code.__path__[0], 'iter_' + iter_path)
	if os.path.isfile(iter_path):
		file = open(iter_path, 'rb')
		ITER = pickle.load(file)
		file.close()
	else:
		ITER = [ i for i in range(len(data))]
		random.shuffle(ITER)
		file = open(iter_path, 'wb')
		pickle.dump(ITER, file)
		file.close()
	# ------------------------------------------------------------

	for j,i in enumerate(ITER):
		ide, text, hs = data.loc[i, 'id'], data.loc[i, 'text '], data.loc[i, tsk]
		if j >= ini and j < fin:
			VAL_DATA.append((ide, text, hs))
		else:
			DATA.append((ide, text, hs))
		bar.update()
	del data

	I = [i for i,_, _ in DATA]
	X = [x for _,x, _ in DATA]
	Y = [y for _,_, y in DATA]
	Ci, Cx, Cy = pd.Series(I), pd.Series(X), pd.Series(Y)
	del X
	del I
	del Y
	result = pd.concat([Ci, Cx, Cy], axis=1)
	result.to_csv(newR1, sep='\t', header=('id', 'txt', 'y'), index=False)
	del Cx
	del Ci
	del Cy
	del result

	if len(VAL_DATA) > 0:
		I = [i for i, _, _ in VAL_DATA]
		X = [x for _, x, _ in VAL_DATA]
		Y = [y for _, _, y in VAL_DATA]
		Ci, Cx, Cy = pd.Series(I), pd.Series(X), pd.Series(Y)
		del X
		del Y
		del I
		result = pd.concat([Ci, Cx, Cy], axis=1)
		result.to_csv(newR2, sep='\t', header=('id', 'txt', 'y'), index=False)
		del Cx
		del Cy
		del Ci
		del result

		return newR1, newR2
	else:
		return newR1

def order_as_reference(reference, file_system):
	sys_dict = {}

	print('# Ordering', colorizar(os.path.basename(file_system)), 'by reference', colorizar(os.path.basename(reference)))
	with open(file_system, 'r') as file:
		for lines in file.readlines():
			line = lines.split('\t')
			if line[0] == 'id':
				continue
			sys_dict.update({line[0]:int(line[1])})
	
	OUT = open(file_system, 'w')
	RET = []

	with open(reference, 'r') as file:
		for lines in file.readlines():
			line = lines.split('\t')

			my_label = None
			try:
				my_label = sys_dict[line[0]]
			except:
				print(headerizar('ERROR::'),
				'The ID on reference file difers from system file, the final file will contain errors')
				my_label = 0
			OUT.write(line[0] + '\t' + re.sub('\n', '', line[1]) + '\t' + str(my_label) + '\n')
	OUT.close()
	print('# Done!! :)')