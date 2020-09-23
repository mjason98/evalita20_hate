from collections import Counter
import numpy as np
import time
from random import shuffle
from math import log
import pickle

def creating_vocab(corpus):
	vocab = Counter()
	
	for line in corpus:
		tokens = line.strip().split()
		vocab.update(tokens)
	
	return { word:(i, frec) for i, (word, frec) in enumerate(vocab.items()) }

def make_coocurrence_matrix(vocab, corpus, window_size=5, min_count=None):
	vocab_len = len(vocab)
	id2word = { i:word for i, word in enumerate(vocab) }
	coocurrence = {}
	
	for co_i, line in enumerate(corpus):
		#Tokenizar una linea
		tokens = line.strip().split()
		tokens_ids = [ vocab[word][0]  for word in tokens ]
		
		#Extraer y calcular Xij para cada palabra en la ventana
		for center_i, center_id in enumerate(tokens_ids):
			context_ids = tokens_ids[max(0, center_i - window_size ) : center_i]
			context_len = len( context_ids )
			
			for word_pos, word_id in enumerate(context_ids):
				#incremento para Xij
				distance = context_len - word_pos
				inc = 1.0 / float(distance)
				
				coo_1 = coocurrence.get((center_id, word_id))
				if coo_1 == None:
					coo_1 = 0.0
				coo_2 = coocurrence.get((word_id, center_id))
				if coo_2 == None:
					coo_2 = 0.0
				
				coo_1 += inc
				coo_2 += inc
				
				dic = {(center_id, word_id):coo_1, (word_id, center_id):coo_2}
				coocurrence.update(dic)
	
	for word_1, word_2 in coocurrence:
		if min_count != None:
			if vocab[ id2word[word_1] ][1] < min_count:
				continue
			elif vocab[ id2word[word_2] ][1] < min_count:
				continue
		yield word_1, word_2, coocurrence[(word_1, word_2)]
		

def train_glove(vocab, coocurrence, iterations=25, vector_size=100, pre_load=False):
	#Entrenando el modelo
	vocab_len = len(vocab)
	#Parametros a entrenar
	W = np.random.rand(vocab_len * 2, vector_size) / float(vector_size + 1)
	bias = (np.random.rand(vocab_len * 2) - 0.5) / float(vector_size + 1)
	

	#Pre-Load if was saved
	if pre_load:
		w_file = open('tmp_W', 'rb')
		b_file = open('tmp_b', 'rb')	
		W = pickle.load(w_file)
		bias = pickle.load(b_file)
		w_file.close()
		b_file.close()


	#Gradientes por AdaGrad
	grad_square = np.ones((vocab_len * 2, vector_size), dtype=np.float64)
	grad_square_bias = np.ones(vocab_len * 2, dtype=np.float64)
	
	data = [( W[main_i] , W[context_i + vocab_len] ,
				  bias[main_i : main_i + 1 ],
				  bias[context_i + vocab_len : context_i + vocab_len+1 ],
				  grad_square[main_i], grad_square[context_i + vocab_len],
				  grad_square_bias[main_i:main_i+1],
				  grad_square_bias[context_i + vocab_len : context_i + vocab_len+1 ],
				  coocurrens) 
				  for main_i, context_i, coocurrens in coocurrence ]
	
	for i in range(iterations):
		print('\n\t#Begin iteration %d:'%(i))
		tiempo = time.time()
		costo = run_iter(vocab, data)
		tiempo = time.time() - tiempo
		
		w_file = open('tmp_W', 'wb')
		b_file = open('tmp_b', 'wb')	
		pickle.dump(W, w_file)
		pickle.dump(bias, b_file)
		w_file.close()
		b_file.close()
		
		print('\t# Done! loss: %.5f time: %.2f' %(costo, tiempo))

	return W

def run_iter(vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):
	#Corriendo una iteraccion del entrenamiento
	#
	# ( v_i, v_j, b_i, b_j,
	#   W_ada_i, W_ada_j, b_ada_i, b_ada_j,
	#   Xij )
	#
	gloval_cost = 0.0
	shuffle(data)
	
	for (v_i, v_j, b_i, b_j, W_ada_i, W_ada_j, b_ada_i, b_ada_j, Xij) in data:
		#f(Xij) computing
		weight = (Xij / x_max) ** alpha if Xij < x_max else 1
		#   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
		cost_inner = (v_i.dot(v_j)) + b_i[0] + b_j[0] -  log(Xij)
		#   $$ J = f(X_{ij}) (J')^2 $$
		cost = weight * (cost_inner ** 2)
		
		gloval_cost += 0.5 * cost
		#Calcular el gradiente
		grad_main     = weight * cost_inner * v_j
		grad_context  = weight * cost_inner * v_i
		
		grad_bias_main = weight * cost_inner
		grad_bias_context = weight * cost_inner
		
		#Adaptative updates
		v_i -= (learning_rate * grad_main / np.sqrt( W_ada_i ))
		v_j -= (learning_rate * grad_context / np.sqrt( W_ada_j ))
		
		b_i -= (learning_rate * grad_bias_main / np.sqrt( b_ada_i ))
		b_j -= (learning_rate * grad_bias_context / np.sqrt( b_ada_j ))
		
		#Update square gradients
		W_ada_i += np.square(grad_main)
		W_ada_j += np.square(grad_context)
		b_ada_i += grad_bias_main ** 2
		b_ada_j += grad_bias_context ** 2
		
	return gloval_cost
	
def save_model(W, vocab, path, header=True):
	id2word = { i:word for i, word in enumerate(vocab) }
	with open(path, 'w', encoding='utf-8') as vector_f:
		if header:
			vector_f.write( str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
		for i in range( len (vocab) ):
			vector_f.write( id2word[i] )
			for j in range (W.shape[1]):
				vector_f.write(' ' + str(W[i][j]))
			vector_f.write('\n')

def start(DIR_NAME, SAVE_PATH):
	# DIR_NAME = 'HAHA_corpus.txt'
	
	print ('# Starting GloVe with ', DIR_NAME)
	
	print ('\n# Creating Vocabulary...')
	with open(DIR_NAME, 'r', encoding='utf-8' ) as fop:
		vocab = creating_vocab(fop)
	print ('# Done!... Vocabulary size: ', len(vocab))
	
	print('\n# Creating the coocurrence matrix...')
	with open(DIR_NAME, 'r', encoding='utf-8' ) as fop:
		coocurrence = list ( make_coocurrence_matrix(vocab, fop) )
	fop.close()
	print('# Done!... Matrix     size: ', len(coocurrence))
	
	itera, vs = 400, 100
	print('\n# Training with', itera, 'iterations and', vs, 'vector size!')
	W = train_glove(vocab, coocurrence, iterations=itera, vector_size=vs)
	print('# Done!.. Saving the model')
	save_model(W, vocab, SAVE_PATH, header=False)
	print('# Done!!! Good Bye ')


if __name__ == '__main__':
	DIR_NAME = '/DATA/work_space/6-IA-ML-DL-RL/2-EvalIta2020/data/_corpus.txt'
	SAVE_PATH = 'glove50l.txt'
	start(DIR_NAME, SAVE_PATH)