import sys
import os
import argparse
import pickle

from code.preprocesing import EVALITA2018_HATE, makePrepData, order_as_reference, EVALITA2020_HATE
from code.utils import colorizar
from code.model import *
from code.ig_feature import calculate_IG, selector_list_featurizer, create_bert_features, recarge_bert, delete_bert

DATA_PATH      = 'data/haspeede2_dev_taskAB.tsv'
EMBEDING_PATH  = 'data/it300.vec'
DATA_PRED_PATH = 'data/haspeede2_test_taskAB-tweets.tsv'
# DATA_PRED_PATH = 'data/haspeede2-test_taskAB-news.tsv'

F_VALIDATION   = 5
SEQ_LENG       = 32 #40 #30
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

	if not os.path.isdir('data'):
		os.mkdir('data')
	if not os.path.isdir('models'):
		os.mkdir('models')
	if not os.path.isdir('photos'):
		os.mkdir('photos')

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

def delete_temporal_files():
	file_list = ['test.tsv', 'iglist', 'params.prm', 'test_prep.csv','train_prep.csv','dev_prep.csv', 'train.tsv', 'dev.tsv', 'test.tsv', 'train_feat.tsv', 'test_feat.tsv', 'dev_feat.tsv', 'bert_train_feat.tsv', 'bert_test_feat.tsv', 'bert_dev_feat.tsv']

	print ('# Deleting Temporal Files')
	for f in file_list:
		file = os.path.join('data', f)
		if os.path.isfile(file):
			os.remove(file)

def prep_features(read_write_paths, first_path=None, bert=False):
	if bert:
		recarge_bert()

	# Calculate the important words for a first time if is needed
	ig = None
	if not os.path.isfile('data/iglist'):
		if first_path is None:
			print ('# ERROR:: ig fetures need a train data file in the first calculation')
			assert (first_path != None)

		print ('# First IG list creation')
		data_raw = pd.read_csv(first_path, sep='\t')
		ig = calculate_IG(data_raw, save_path='data/iglist')
	else:
		file = open('data/iglist', 'rb')
		ig = pickle.load(file)
		file.close()

	for r, w in read_write_paths:
		data_raw = pd.read_csv(r, sep='\t')
		selector_list_featurizer(ig, data_raw, w)
		if bert:
			w_bert = os.path.join(os.path.dirname(w), 'bert_' + os.path.basename(w))
			create_bert_features(data_raw, w_bert, max_length=SEQ_LENG)
	
	if bert:
		delete_bert()

def train_model(train_path, eval_path):
	save_path = 'models/hate_model.pt'

	# return save_path

	VOC = {}
	model = makeModel(LAYER1, SEQ_LENG, DRP,
			Enb = generate_dictionary_from_embedding(EMBEDING_PATH, VOC, logs=True),
			arquitectura='la')
	# initModel(model)

	train_path = makePrepData(train_path, vocab=VOC, seq_len=SEQ_LENG, name_only=preproces, separete_sents=False, feature_list=['data/train_feat.tsv', 'data/bert_train_feat.tsv'])
	eval_path  = makePrepData(eval_path,  vocab=VOC, seq_len=SEQ_LENG, name_only=preproces, separete_sents=False, feature_list=['data/dev_feat.tsv', 'data/bert_dev_feat.tsv'])

	del VOC

	train_data, train_load = makeData(train_path, BATCH, second=False)
	val_data, val_load     = makeData(eval_path, BATCH, second=False)

	board = TorchBoard()
	total_acc_tr, total_acc_val = Train(model, train_load, val_load, lr=LR, 
								  epochs=EPOCH, board=board, save_path=save_path)
	board.show('photos/model' + str(1) + '.png', plot_smood=True)
	print ('# END Model: train %s | val %s' % (total_acc_tr.max(), total_acc_val.max()))

	del board
	del val_load
	del train_load
	del train_data
	del val_data
	del model

	return save_path

def predict_test_data(pt_path, save_name='data/prediction.tsv'):
	global DATA_PRED_PATH

	if not os.path.isfile(DATA_PRED_PATH):
		print ('# Data', colorizar(os.path.basename(DATA_PRED_PATH)), 'not found!')
		return

	test_path = EVALITA2020_HATE(DATA_PRED_PATH, validation=(0,1), names=['data/test.tsv', ''], not_label=True)
	prep_features([(test_path, 'data/test_feat.tsv')], bert=True)

	VOC = {}
	model = makeModel(LAYER1, SEQ_LENG, DRP,
			Enb = generate_dictionary_from_embedding(EMBEDING_PATH, VOC, logs=True),
			arquitectura='la')

	test_path = makePrepData(test_path, vocab=VOC, seq_len=SEQ_LENG, name_only=preproces, separete_sents=False, feature_list=['data/test_feat.tsv', 'data/bert_test_feat.tsv'])
	
	print ('# Making Predictions as', colorizar(os.path.basename(save_name)))

	test, test_l = makeData(test_path, BATCH, second=False, shuffle=False)
	EvalData(model, test_l, filepath=pt_path,
			save_path=save_name, 
			header=('id', 'hs', ''), label_prediction=True)

	del test_l
	del test
	del VOC
	del model

	return save_name

def main():
	global DATA_PATH
	calculatePreprocesing()

	train_path, eval_path = EVALITA2020_HATE(DATA_PATH, validation=(0,10), task=TASK)

	prep_features([(train_path, 'data/train_feat.tsv'), (eval_path, 'data/dev_feat.tsv')], first_path=train_path, bert=True)

	model_path = train_model(train_path, eval_path)

	predi_path = predict_test_data(model_path, save_name=os.path.join('send', os.path.basename(DATA_PRED_PATH)))

	order_as_reference(DATA_PRED_PATH, predi_path)
	
	# Opcional
	delete_temporal_files()

if __name__ == '__main__':
	check_params(arg=sys.argv[1:])
	main()