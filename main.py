'''
	********************************************
			EVALITA 2020 Hatespreed
	********************************************
	=> main.py is the main code to run.
	=> This attempt to preprocess the data, train a Deep model an predict labels
	   in the task evalita 2020 hatespreed
'''

import sys
import os
import argparse
import pickle

from code.preprocesing import *
from code.utils import *
from code.model import *

DATA_PATH      = 'data/haspeede2_dev_taskAB.tsv'
DATA_PATH      = 'data/EvalIta-HateSpeech2018/FB-folder-20200907T154715Z-001/FB-folder/FB-train/haspeede_FB-train.tsv'
# EMBEDING_PATH  = 'data/it300.vec'
DATA_PRED_PATH = 'data/EvalIta-HateSpeech2018/FB-folder-20200907T154715Z-001/FB-folder/FB-test/haspeede_FB-test.tsv'
DATA_PRED_PATH = ''

F_VALIDATION   = 5
SEQ_LENG       = 20 #40 #30
BATCH   	   = 128
LAYER1		   = 128 #100 #70 #50
LR   		   = 0.01
EPOCH  		   = 20 # 20
DRP            = 0.3
TASK 		   = 'hs'
new_exp        = False
preproces      = False

FIRST_ARQ      = 'conv'
merger_list    = ['lstm_s', 'mann_s']

def check_params(arg=None):
	global F_VALIDATION
	global BATCH
	global LR
	global DRP
	global LAYER1
	global EPOCH
	global SEQ_LENG

	parse = argparse.ArgumentParser(description='EVALITA2020 Hate')
	parse.add_argument('-l', '--learning_rate', help='The learnng rate to the optimizer', 
					   required=False, default=LR)
	parse.add_argument('-b', '--batchs', help='Amount of batchs', 
					   required=False, default=BATCH)
	parse.add_argument('-e', '--epochs', help='Amount of epochs', 
					   required=False, default=EPOCH)
	parse.add_argument('-q', '--sequence_size', help='the maximun sequence size', 
					   required=False, default=SEQ_LENG)
	parse.add_argument('-d', '--dropout', help='the dropout', 
					   required=False, default=DRP)
	parse.add_argument('-f', '--valid', help='F-validation number for F', 
					   required=False, default=F_VALIDATION)
	parse.add_argument('-n', '--hidden_size', help='the size of the hidden layer in the first Bi-Lstm', required=False, default=LAYER1)
   
	returns = parse.parse_args(arg)

	LR    = float(returns.learning_rate)
	BATCH = int(returns.batchs)
	EPOCH = int(returns.epochs)
	SEQ_LENG = int(returns.sequence_size)
	DRP    = float(returns.dropout)
	LAYER1 = int(returns.hidden_size)
	F_validation = int(returns.valid)

def calculatePreprocesing():
	global preproces

	preproces = False
	if os.path.isfile('data/params.prm'):
		f = open('data/params.prm', 'rb')
		tup = pickle.load(f)
		if tup[0] == EMBEDING_PATH and tup[1] == SEQ_LENG and tup[2] == DATA_PATH:
			preproces = True
		f.close()

	tup = (EMBEDING_PATH, SEQ_LENG, DATA_PATH)
	f = open('data/params.prm', 'wb')
	pickle.dump(tup, f)
	f.close()

def train_sentence_encoder(train_path, eval_path):
	global DATA_PATH

	save_path = 'models/hate_sentenc.pt'
	# First Block: Training the sentence encoder
	VOC = {}
	model = makeModel(LAYER1, SEQ_LENG, DRP,
			Enb = generate_dictionary_from_embedding(EMBEDING_PATH, VOC, logs=True),
			arquitectura=FIRST_ARQ) # siamlstm # best conv

	train_path = makePrepData(train_path, vocab=VOC, seq_len=SEQ_LENG, name_only=preproces)
	eval_path  = makePrepData(eval_path,  vocab=VOC, seq_len=SEQ_LENG, name_only=preproces)

	initModel(model)

	train_data, train_load = makeData(train_path, BATCH, second=False)
	val_data, val_load     = makeData(eval_path, BATCH, second=False)

	board = TorchBoard()
	total_acc_tr, total_acc_val = Train(model, train_load, val_load, lr=LR, 
									epochs=EPOCH, board=board, save_path=save_path)
	board.show('photos/model_enc' + str(1) + '.png')
	print ('# END Sentence Encoder: train %s | val %s' % (total_acc_tr.max(), total_acc_val.max()))

	del board
	del val_load
	del train_load
	del train_data
	del val_data
	
	train_data, train_load = makeData(train_path, BATCH, second=False, shuffle=False)
	val_data, val_load     = makeData(eval_path,  BATCH, second=False, shuffle=False)

	# Second Block: Mixing the sentence vectors to train the next model
	EvalDataVec(model, train_load, save_path, out_path='data/encoder_train.csv', leng_fized=19)
	EvalDataVec(model, val_load,  save_path, out_path='data/encoder_eval.csv'  , leng_fized=19)

	del model
	del VOC
	del val_load
	del train_load
	del train_data
	del val_data
	# clear_preprocessing()
	# clear_free()

def train_decoder_sentence():
	# Multi model Decoder
	global merger_list
	for arq in merger_list:
		VOC     = {}
		total_acc_val, total_acc_tr = 0, 0
		model = makeModel(LAYER1, SEQ_LENG, DRP, arquitectura=arq,
				Enb = generate_dictionary_from_embedding(EMBEDING_PATH, VOC,
					  logs=True, message_logs='with arq. '+arq))
		del VOC
		initModel(model)
		
		train_path, val_path = 'data/encoder_train.csv', 'data/encoder_eval.csv'
		test_path = 'data/encoder_test.csv'

		train_data, train_load = makeData(train_path, BATCH, second=True)
		val_data, val_load     = makeData(val_path, BATCH, second=True)
		board = TorchBoard()

		# Training the Decoder Arquitecture
		total_acc_tr, total_acc_val = Train(model, train_load, val_load, 
										lr=LR, epochs=EPOCH, board=board,
										save_path='models/hate_dec_'+ arq +'.pt')
		board.show('photos/model_dec'+ arq +'.png')
		print ('# END: train %s | val %s' % (total_acc_tr.max(), total_acc_val.max()))
		del board

		# Saving vector representation per model
		EvalData(model, train_load, filepath='models/hate_dec_'+ arq +'.pt', save_path='data/sage_train' + arq + '.csv')
		EvalData(model, val_load, filepath='models/hate_dec_'+ arq +'.pt', save_path='data/sage_eval' + arq + '.csv')

		del val_load
		del train_load
		del train_data
		del val_data

		# Prediction Section if test data exist
		if os.path.isfile(test_path):
			test_data, test_load = makeData(test_path, BATCH, second=True)
			EvalData(model, test_load, filepath='models/hate_dec_'+ arq +'.pt', save_path='data/sage_test' + arq + '.csv')

			del test_data
			del test_load
		del model

	# Making the final data to merge vector representation of all models
	path_list_train = []
	path_list_eval  = []
	path_list_test  = []
	for arq in merger_list:
		path_list_train.append('data/sage_train' + arq + '.csv')
		path_list_eval.append('data/sage_eval' + arq + '.csv')
		path_list_test.append('data/sage_test' + arq + '.csv')

	makeSageData(path_list_train, 'data/train_sage.csv')
	makeSageData(path_list_eval, 'data/dev_sage.csv')
	
	if os.path.isfile(test_path):
		makeSageData(path_list_test, 'data/test_sage.csv')

def evalution_predictions():
	global DATA_PRED_PATH

	if not os.path.isfile(DATA_PRED_PATH):
		return

	test_path = EVALITA2018_HATE(DATA_PRED_PATH, validation=(0,1), names=['data/test.tsv', ''])
	
	# First model evaluation
	
	VOC, save_path = {}, 'models/hate_sentenc.pt'
	model = makeModel(LAYER1, SEQ_LENG, DRP,
			Enb = generate_dictionary_from_embedding(EMBEDING_PATH, VOC, logs=True),
			arquitectura=FIRST_ARQ)
	
	test_path = makePrepData(test_path, vocab=VOC, seq_len=SEQ_LENG, name_only=preproces)
	test_data, test_load = makeData(test_path, BATCH, second=False, shuffle=False)
	EvalDataVec(model, test_load, save_path, out_path='data/encoder_test.csv', leng_fized=19)

	del test_load
	del test_data
	del VOC
	del model

def main():
	global DATA_PATH
	calculatePreprocesing()

	train_path, eval_path = EVALITA2020_HATE(DATA_PATH, validation=(0,10)) #2018
	
	# Sentece Encoder Training Process
	train_sentence_encoder(train_path, eval_path)
	evalution_predictions()
	
	# Sentece Decoder Training Process
	train_decoder_sentence()

	# Final Layer Training
	runSageTrain(BATCH)
	# order_as_reference(DATA_PRED_PATH, 'data/prediction.tsv')

	# Delete Temporal files

def main2():
	global DATA_PATH
	# calculatePreprocesing()

	train_path, eval_path = EVALITA2020_HATE(DATA_PATH, validation=(0,10), tsk=TASK)


if __name__ == '__main__':
	check_params(arg=sys.argv[1:])
	main()
	# main2()