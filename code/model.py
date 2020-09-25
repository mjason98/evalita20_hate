import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd
import math
from .preprocesing import getMyDict
from torch.utils.data import Dataset, DataLoader
from .utils import *
import sklearn.metrics as met
import pickle

def initW(m):
	if type(m) == nn.Linear:
		torch.nn.init.xavier_uniform_(m.weight)
		#m.bias.data.zero()

def initModel(model):
	model.apply(initW)

def loadModel(model, path):
	model.load(path)

def saveModel(model, path):
	model.save(path)

def makeWeight(y, alpha=0.8):
	assert alpha > 1 or alpha < 0
	with torch.no_grad():
		w = alpha * (y == 1).float() +  (1.-alpha) * (y == 0).float()
	return w

def Train(model, data_train, data_val=None, save_path='models/hate.pt', epochs=20, lr=0.001, board=None):
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	model.train()
	ret_val, ret_tr = [], []

	for e in range(epochs):
		barr = StatusBar(len(data_train), title=headerizar('# Epoch %s/%s'%(e+1, epochs)))
		total_s, total_l = 0,0
		total_acc = 0
		for data in data_train:
			optim.zero_grad()

			y_hat, y_ml = model(data['x1'], data['f'])
			y     = data['y']

			loss1 = model.criterion1(y_hat, y)
			loss2 = model.criterion2(y_ml,  y)

			loss1 = (data['i'] >= 0) * loss1
			loss2 = (data['i'] <  0) * loss2

			loss = 0.8*loss1.mean() + 0.2*loss2.mean()
			
			loss.backward()
			optim.step()
			
			total_s   += y_hat.shape[0]
			total_l   += loss.item() * y_hat.shape[0]
			total_acc += (y_hat.argmax(dim=1) == y).sum().item()
			barr.update(loss=[('loss', total_l/total_s)])

		total_acc /= total_s
		ret_tr.append(total_acc)
		if board is not None:
			res = board.update('train', total_acc, getBest=True)
			if data_val is None and res:
				saveModel(model, save_path)
		if data_val is not None:
			va_acc = Val(model, data_val, metrics=False)
			ret_val.append(va_acc)
			
			res = board.update('test', va_acc, getBest=True)
			if res:
				saveModel(model, save_path)

	if data_val is not None:
		return np.array(ret_tr), np.array(ret_val)
	else:
		return np.array(ret_val)

def Val(model, data, metrics=False):
	model.eval()
	if metrics:
		pred, true = None, None
		with torch.no_grad():
			barr = StatusBar(len(data), title=' Validating:')
			for d in data:
				y_hat, _ = model(data['x1'], data['f'])
				y_hat = y_hat.argmax(dim=1).numpy()

				if true is None:
					true = d['y'].numpy()
					pred = y_hat
				else:
					true = np.concatenate([true, d['y'].numpy()], axis=0)
					pred = np.concatenate([pred, y_hat], axis=0)
				barr.update(loss=[('prec', met.precision_score(true, pred))])
		acc, prec = met.accuracy_score(true, pred), met.precision_score(true, pred)
		rec, f1   = met.recall_score(true, pred), met.f1_score(true, pred)
		model.train()
		return np.array([acc, prec, rec, f1])
	else:
		with torch.no_grad():
			barr = StatusBar(len(data), title=' Validating:')
			v,s = 0,0
			for d in data:
				y_hat,_ = model(d['x1'], d['f'])
				v += (y_hat.argmax(dim=1) == d['y']).sum().item()
				s += y_hat.shape[0]
				barr.update(loss=[('acc', v/s)])
		model.train()
		return v/s

def EvalDataVec(model, data, filepath, out_path='data/taskVev.csv', leng_fized=None):
	loadModel(model, filepath)
	model.eval()

	eps = 1e-9
	F_DATA, LB, FEAT = {}, {}, {}
	vec_leng, max_leng = 0, 0

	if leng_fized is not None:
		max_leng = leng_fized
		print('# Making', headerizar(os.path.basename(out_path)), 'with fixed size', leng_fized)
	else:
		print('# Making', headerizar(os.path.basename(out_path)))

	# Evaluating the data throo the model
	with torch.no_grad():
		barr = StatusBar(len(data), title='# Vectors:')
		for d in data:
			feat  = d['f']
			x     = d['x1']

			y_hat = model(x, feat, ret_vec=True)
			ides, lab  = d['i'], d['y']

			for i in range(y_hat.shape[0]):
				vec = y_hat[i].numpy()
				ide = int(ides[i].item())

				LB.update({ide:lab[i].item()})
				if F_DATA.get(ide) is not None:
					F_DATA[ide].append(vec)
					max_leng = max(max_leng, len(F_DATA[ide]))
				else:
					F_DATA.update({ide:[vec]})
				if FEAT.get(ide) is not None:
					nv = FEAT[ide] + feat[i].numpy()
					FEAT.update({ide:nv})
				else:
					FEAT.update({ide:feat[i].numpy()})
			barr.update()
			
			if vec_leng == 0:
				vec_leng = y_hat.shape[1]
	
	# Merging the data
	barr = StatusBar(len(F_DATA), title='# Merging:')
	DATA = []
	for ide in F_DATA:
		vec = np.concatenate(F_DATA[ide])
		if vec.shape[0] < vec_leng*max_leng:
			vec = np.concatenate([vec, np.zeros(vec_leng*max_leng - vec.shape[0])])
		vec  = ' '.join([str(i) for i in vec])
		lab  = LB[ide]
		
		feat = FEAT[ide]
		feat = feat / (np.sqrt((feat*feat).sum()) + eps)
		feat = ' '.join([str(i) for i in feat])

		DATA.append((ide, vec, lab, feat))
		barr.update()

	del F_DATA
	del LB

	A, B, C, D = pd.Series([i for i, _, _, _ in DATA]), pd.Series([i for _, i, _, _ in DATA]), pd.Series([i for _, _, i, _ in DATA]), pd.Series([i for _, _, _, i in DATA])
	result = pd.concat([A,B,C, D], axis=1)
	result.to_csv(out_path, header=('id', 'vectors', 'hs', 'feat'), index=False)
	del DATA
	del A
	del B
	del C 
	del D
	del result
	print ('# End: vector size %d and max vectors %d'% (vec_leng,max_leng))

def EvalData(model, data, filepath=None, save_path=None, header=('id', 'hs', 'vec'), label_prediction=False):
	assert len(header) == 3
	if filepath is not None:
		loadModel(model, filepath)
	model.eval()

	RET = []
	with torch.no_grad():
		barr = StatusBar(len(data), title='# Eval:')
		for d in data:
			ides  = d['i']
			if label_prediction:
				# Predict the final label
				
				y_hat, _ = model(d['x1'], d['f'])
				y_hat  = y_hat.argmax(dim=-1)
				for i in range(y_hat.shape[0]):
					RET.append((ides[i].item(), y_hat[i].item()))
			else:
				# Predict the vector representation
				
				y_hat, y = model(d['x1'], d['f'], ret_vec=True), d['y']
				for i in range(y_hat.shape[0]):
					RET.append((ides[i].item(), y[i].item(),' '.join([str(v.item()) for v in y_hat[i]])))
			barr.update()
		
	if save_path is None:
		return RET

	if label_prediction:
		header = (header[0], header[1])
		A, B = pd.Series([i for i, _ in RET]), pd.Series([i for _, i in RET])
		result = pd.concat([A, B], axis=1)

		if save_path[-4:] == '.tsv':
			result.to_csv(save_path, sep='\t', header=header, index=False)
		else:
			result.to_csv(save_path, header=header, index=False)

		del RET
		del A
		del B
		del result
	else:
		A, B, C = pd.Series([i for i, _, _ in RET]), pd.Series([i for _, i, _ in RET]), pd.Series([i for _, _, i in RET])
		result = pd.concat([A, B, C], axis=1)

		if save_path[-4:] == '.tsv':
			result.to_csv(save_path, sep='\t', header=header, index=False)
		else:
			result.to_csv(save_path, header=header, index=False)

		del RET
		del A
		del B
		del C
		del result


def makeSageData(data_path_list, new_name, header=('id', 'hs', 'vec')):
	# Here all models vector representation 
	# its merged into a new vector and saved
	# in a new data to train a feedforward net
	# named Sage

	print('# Making', os.path.basename(new_name), end=' ')
	TR = {}
	for t in data_path_list:
		data = pd.read_csv(t, encoding='utf-8')
		for i in range(len(data)):
			ide, hs, vec = data.loc[i, header[0]], data.loc[i, header[1]], data.loc[i, header[2]]
			if TR.get(ide) is None:
				TR.update({ide:[ide, hs, vec]})
			else:
				TR[ide][2] += ' ' + vec
		del data
	TR  = [TR[i] for i in TR]

	A, B, C = pd.Series([i for i, _, _ in TR]), pd.Series([i for _, i, _ in TR]), pd.Series([i for _, _, i in TR])
	result = pd.concat([A, B, C], axis=1)
	result.to_csv(new_name, header=header, index=False)
	del TR
	del A
	del B
	del C
	del result

	print(' Done!')

# =========================================================================
class AttFeat(nn.Module):
	def __init__(self, feat_size, proj_size):
		super(AttFeat, self).__init__()
		self.size = feat_size
		self.proj = proj_size
		self.R    = nn.Sequential(nn.Linear(feat_size, proj_size), nn.LeakyReLU())
		self.A    = nn.Linear(proj_size, 1, bias=False)

	def forward(self, X):
		ri, ai = [], []
		for v in X:
			r = self.R(v)
			ai.append(self.A(torch.tanh(r)))
			ri.append(r.reshape(-1,1,self.proj))
		a = torch.cat(ai, dim=-1)
		a = F.softmax(a,  dim=-1).reshape(-1,len(X), 1)
		r = torch.cat(ri, dim=1)
		s = (a*r).sum(dim=1)
		return F.relu(s)

class AttBlock(nn.Module):
	def __init__(self, len_seq, input_size, hidden_size, output_size, batch=64, dropout=0.0, num_heads=1):
		super(AttBlock, self).__init__()
		
		self.cell     = nn.LSTMCell(input_size*3, hidden_size)
		self.densa1   = nn.Linear(hidden_size,len_seq)
		self.densa2   = nn.Linear(hidden_size,len_seq)
		self.densa3   = nn.Linear(hidden_size,len_seq)
		self.drop     = nn.Dropout(dropout)
		
		self.hidden_size = hidden_size
		self.batch       = batch
		self.len_seq     = len_seq
		self.outp        = output_size
	
	def init_weights(self):
		self.densa.weight.data.uniform_(-1.0,1.0)
		self.densa.bias.data.zero_()
	
	def init_hidden(self, bath=None):
		if bath ==  None:
			return (torch.zeros(self.batch, self.hidden_size),
					torch.zeros(self.batch, self.hidden_size))
		else:
			return (torch.zeros(bath, self.hidden_size),
					torch.zeros(bath, self.hidden_size))
	
	def forward(self, x, hc, mati=False):
		#x.shape = (batch, len_seq, input_size)
		out, ploto = [], []
		h_i, c_i = hc
		for i in range(self.outp):
			#attention mechanism head_1
			alp1 = self.densa1(h_i)
			alp1_soft = F.softmax(alp1,dim=1)
			alg1 = alp1_soft.reshape(-1, self.len_seq, 1)
			alg1_dr = self.drop(alg1)
			x_i1 = x * alg1_dr
			v1   = torch.sum(x_i1,dim=1)

			#attention mechanism head_2
			alp2 = self.densa2(h_i)
			alp2_soft = F.softmax(alp2,dim=1)
			alg2 = alp2_soft.reshape(-1, self.len_seq, 1)
			alg2_dr = self.drop(alg2)
			x_i2 = x * alg2_dr
			v2   = torch.sum(x_i2,dim=1)

			#attention mechanism head_3
			alp3 = self.densa3(h_i)
			alp3_soft = F.softmax(alp3,dim=1)
			alg3 = alp3_soft.reshape(-1, self.len_seq, 1)
			alg3_dr = self.drop(alg3)
			x_i3 = x * alg3_dr
			v3   = torch.sum(x_i3,dim=1)
			
			v = torch.cat([v1, v2, v3],dim=-1)
			#lstm cell after attention
			h_i, c_i = self.cell(v, (h_i, c_i))
			out.append(h_i.reshape(x.shape[0],1,self.hidden_size))
			
			if mati == True:
				ploto.append(alp1_soft)
		out_c = torch.cat(out, dim=1)
		if mati == True:
			return out_c, ploto
		return out_c

class AttRnnBlk(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, dropout=0.0, num_heads=1):
		super(AttRnnBlk, self).__init__()
		
		self.eps = 1e-6
		self.cell     = nn.LSTMCell(input_size, hidden_size)
		self.densa    = []
		self.drop     = nn.Dropout(dropout)
		self.sumary   = nn.Linear(input_size*num_heads, input_size)
		self.tanh     = nn.Tanh()
		
		self.hidden_size = hidden_size
		self.outp        = output_size
		self.num_heads   = num_heads

		for _ in range(self.num_heads):
			self.densa.append(nn.Linear(hidden_size,input_size))

		self.init_weights()
	
	def init_weights(self):
		for i in range(self.num_heads):
			self.densa[i].weight.data.uniform_(-1.0,1.0)
			self.densa[i].bias.data.zero_()
	
	def init_hidden(self, bath):
		return (torch.zeros(bath, self.hidden_size),
				torch.zeros(bath, self.hidden_size))
	
	def forward(self, x, hc):
		#x.shape = (batch, len_seq, input_size)
		out, ploto = [], []
		h_i, c_i = hc

		x_sqrt = x.pow(2).sum(dim=2).sqrt()
		for i in range(self.outp):
			v = []
			for densa in self.densa:
				key = densa(h_i).reshape(x.shape[0], 1, x.shape[2])
				inner_dot = (key * x).sum(dim=2)
				k_sq      = key.pow(2).sum(dim=2).sqrt() + x_sqrt
				w         = inner_dot / (k_sq  + self.eps)
				w         = F.normalize(w, dim=1)
				w         = w.reshape(x.shape[0], x.shape[1], 1)
				val       = (w * x).sum(dim=1)
				v.append(val)
			
			v = torch.cat(v, dim=-1)
			v = self.tanh(self.sumary(v))
			#lstm cell after attention
			h_i, c_i = self.cell(v, (h_i, c_i))
			out.append(h_i.reshape(x.shape[0],1,self.hidden_size))
		out_c = torch.cat(out, dim=1)
		return out_c

class MAN(nn.Module):
	"""docstring for MAN"""
	def __init__(self, in_size, hidd_size, memory_size=50, alpha_sig=-0.4, gamma=0.3):
		super(MAN, self).__init__()
		self.memory_size = memory_size
		self.memory_vector_size = hidd_size

		self.controler = nn.LSTMCell(in_size, hidd_size)
		self.M = np.zeros((self.memory_size, self.memory_vector_size), dtype=np.float32)
		self.gate = 1. / (1. + math.exp(-alpha_sig))
		self.gamma = gamma

	def init_hidden(self, batch=1):
		return (torch.zeros(batch, self.memory_vector_size),
				torch.zeros(batch, self.memory_vector_size))

	def read_head(self, K, prev_M_t):
		inner_dot = K @ prev_M_t.T
		k_sq = K.pow(2).sum(dim=-1, keepdim=True).sqrt()
		m_sq = prev_M_t.pow(2).sum(dim=-1, keepdim=True).sqrt()
		norm_dot = k_sq @ m_sq.T
		coss_sim = inner_dot / (norm_dot + 1e-8)
		coss_sim = F.softmax(coss_sim, dim=-1)
		return coss_sim

	def write(self, k, w_r, w_u, prev_r, prev_M_t):
		with torch.no_grad():
			prev_w_u = torch.from_numpy(w_u).detach()
			prev_w_r = torch.from_numpy(prev_r).detach()
			key      = k.detach()
			w_lu = torch.argmin(prev_w_u, dim=-1).long()
			w_lu = (1. - self.gate) * F.one_hot(w_lu, num_classes=self.memory_size).float()
			w_w  = self.gate*prev_w_r + w_lu

			# usar prev_M_T para normalizar si da error
			k_w  = torch.einsum('bi,bj->bij', w_w, key).sum(dim=0)
			k_w  = F.normalize(prev_M_t + k_w, dim=-1).numpy()
			self.M[:,:] = k_w[:,:]

			prev_r[:,:] = w_r.detach().numpy()[:,:]
			new_u    = F.normalize(self.gamma*prev_w_u + w_r + w_w, dim=-1).numpy()
			w_u[:,:]   = new_u[:,:]

			temp_u = torch.from_numpy

	# ver si es mas optimo calcular el de escribir para el siguiente
	def forward(self, X, back=False):
		in_shape = X.shape 
		out      = []
		h, c     = self.init_hidden(in_shape[0])
		
		w_u      = np.zeros((in_shape[0], self.memory_size))
		prev_w_r = np.zeros((in_shape[0], self.memory_size))

		iterator = [i for i in range(in_shape[1])]
		if back:
			iterator.reverse()
		for i in iterator:
			M = torch.from_numpy(self.M).detach()
			
			# Read from the memory
			h, c = self.controler(X[:,i,:], (h, c))
			w_r = self.read_head(h, M)
			read_i = w_r @ M
			to_save = torch.cat([h, read_i], dim=-1).reshape(in_shape[0],1,self.memory_vector_size*2)
			out.append(to_save)

			# Write to the memory
			self.write(h, w_r, w_u, prev_w_r, M)

		return torch.cat(out, dim=1)

class MaxLayer(nn.Module):
	def __init__(self):
		super(MaxLayer, self).__init__()

	def forward(self, X):
		seq_len = X.shape[1]
		x_hat = X.permute((0,2,1))
		x_hat = F.max_pool1d(x_hat,seq_len, stride=1)
		x_hat = x_hat.permute((0,2,1))
		return x_hat.squeeze()

class MeanLayer(nn.Module):
	def __init__(self):
		super(MeanLayer, self).__init__()

	def forward(self, X):
		x_hat = X.mean(dim=1)
		return x_hat.squeeze()

class AddNormLayer(nn.Module):
	def __init__(self):
		super(AddNormLayer, self).__init__()

	def forward(self, X):
		x_hat = F.normalize(X.sum(dim=1), dim=-1)
		return x_hat.squeeze()

class ConvNet(nn.Module):
	def __init__(self, neuron, leng, Embedding=None,feat_size=49, embeding_shape=(13,10), dropout=0.2):
		super(ConvNet, self).__init__()
		self.criterion1 = nn.CrossEntropyLoss()
		self.neu = neuron
		self.leng = leng

		# word embedding
		self.emb_w     = nn.Embedding.from_pretrained(torch.from_numpy(Embedding))#, freeze=False)
		embeding_shape = (Embedding.shape[0], Embedding.shape[1])
		#embeding_shape = (embeding_shape[0], Embedding[0].shape[1])
		
		self.drop = nn.Dropout(dropout)
		self.C1   = nn.Sequential(nn.Conv1d(embeding_shape[1], embeding_shape[1], kernel_size=7, padding=3), nn.ReLU() )
		self.C2   = nn.Sequential(nn.Conv1d(embeding_shape[1], embeding_shape[1], kernel_size=3, padding=1), nn.ReLU() )
		self.C3   = nn.Sequential(nn.Conv1d(embeding_shape[1], embeding_shape[1], kernel_size=5, padding=2), nn.ReLU() )

		self.Layer1  = nn.LSTM(input_size= embeding_shape[1],
						hidden_size= neuron,
						batch_first=True,
						bidirectional=True,
						num_layers=1)

		self.RedLayer = MaxLayer()
		self.Dense1   = nn.Sequential(nn.Linear(neuron*2, feat_size), nn.ReLU())
		self.Dense2   = nn.Linear(130, 2)
		self.FeatRed  = AttFeat(feat_size, 130)

	
	def initHidden(self, batch):
		return (torch.zeros(2,batch, self.neu),
				torch.zeros(2,batch, self.neu))
				

	def forward(self, X, Fe, ret_vec=False):
		x = self.drop(self.emb_w(X))
		x = x.permute(0,2,1)

		f1, f2, f3 = self.C1(x), self.C2(x), self.C3(x)
		x1 = torch.cat([f1, f2, f3], dim=2)
		x1 = F.max_pool1d(x1,kernel_size=3)
		x1 = x1.permute(0,2,1)

		hc1   = self.initHidden(x1.shape[0])
		x1, _ = self.Layer1(x1, hc1)
		
		x = self.RedLayer(x1)
		y = self.Dense1(x)
		y = self.FeatRed([y, Fe])
		
		if ret_vec:
			return y
		y1 = self.Dense2(y).squeeze()
		return y1

	def load(self, path):
		self.load_state_dict(torch.load(path))

	def save(self, path):
		torch.save(self.state_dict(), path) 

class ManNet(nn.Module):
	def __init__(self, hidden_size, len_seq,Embedding=None, feat_size=59, embeding_shape=(13,50), dropout=0.2):
		super(ManNet, self).__init__()
		self.criterion1 = nn.CrossEntropyLoss()
		self.hidden_size = hidden_size

		self.emb_w      = nn.Embedding.from_pretrained(torch.from_numpy(Embedding))#, freeze=False)
		embeding_shape = (Embedding.shape[0], Embedding.shape[1])
		
		self.dro1    = nn.Dropout(dropout)
		self.Layer1  = MAN(embeding_shape[1], hidden_size, memory_size=100)

		self.RedLayer = MeanLayer()
		self.Dense1   = nn.Sequential(nn.Linear(hidden_size*2, hidden_size), nn.ReLU())
		self.Dense2   = nn.Linear(hidden_size + feat_size//2, 2)

		self.FeatRed  = nn.Sequential(nn.Linear(feat_size, feat_size//2), nn.ReLU())

	def forward(self, X, F, ret_vec=False):
		# feature prep
		f = self.FeatRed(F)

		x = self.dro1(self.emb_w(X))
		x = self.Layer1(x)

		x = self.RedLayer(x)
		x = self.Dense1(x)
		y = torch.cat([x, f], dim=-1)
		if ret_vec:
			return y
		y = self.Dense2(y)
		return y.squeeze()

	def load(self, path):
		self.load_state_dict(torch.load(path))

	def save(self, path):
		torch.save(self.state_dict(), path) 

class LstmAtt(nn.Module):
	def __init__(self, hidden_size, len_seq, Embedding, feat_size=49, embeding_shape=(13,50), dropout=0.2):
		super(LstmAtt, self).__init__()
		self.criterion1 = nn.CrossEntropyLoss()#weight=torch.Tensor([0.8,1.2]))
		self.hidden_size = hidden_size

		# word embedding
		self.emb_w   = nn.Embedding.from_pretrained(torch.from_numpy(Embedding))#, freeze=False)
		
		self.dro1    = nn.Dropout(dropout)
		self.Layer1  = nn.LSTM(input_size= Embedding.shape[1],
						hidden_size= hidden_size,
						batch_first=True,
						bidirectional=True,
						num_layers=2)
		self.nora    = nn.BatchNorm1d(hidden_size*2)

		self.RedLayer = MaxLayer() # MeanLayer()
		self.RedLayer = MaxLayer()
		self.Dense1   = nn.Sequential(nn.Linear(hidden_size*2, feat_size), nn.ReLU())
		self.Dense2   = nn.Linear(130, 2)
		self.FeatRed  = AttFeat(feat_size, 130)

	def initHidden(self, batch):
		return (torch.zeros(4,batch, self.hidden_size),
				torch.zeros(4,batch, self.hidden_size)) #,
				#self.Layer2.init_hidden(batch))

	def forward(self, X, Fe, ret_vec=False):
		x1 = self.dro1(self.emb_w(X))

		hc1   = self.initHidden(x1.shape[0])
		x1, _ = self.Layer1(x1, hc1)

		x1 = x1.permute((0,2,1))
		x1 = self.nora(x1)
		x1 = x1.permute((0,2,1))
		
		x = self.RedLayer(x1)
		y = self.Dense1(x)
		y = self.FeatRed([y ,Fe])
		
		if ret_vec:
			return y
		y1 = self.Dense2(y).squeeze()
		return y1

	def load(self, path):
		self.load_state_dict(torch.load(path))

	def save(self, path):
		torch.save(self.state_dict(), path) 

class LstmAtt_S(nn.Module):
	def __init__(self, hidden_size, len_seq, feat_size=49, dropout=0.2):
		super(LstmAtt_S, self).__init__()
		self.criterion1 = nn.CrossEntropyLoss()#weight=torch.Tensor([0.8,1.2]))
		self.hidden_size = hidden_size
		
		self.dro1    = nn.Dropout(dropout)
		# self.Layer1  = nn.LSTM(input_size= 129,
		# 				hidden_size= hidden_size,
		# 				batch_first=True,
		# 				bidirectional=True,
		# 				num_layers=1)
		# self.Layer2  = AttRnnBlk(input_size = 130, #hidden_size*2,
		# 						 output_size=len_seq,
		# 						 hidden_size=hidden_size,
		# 						 num_heads=3,
		# 						 dropout=dropout)
		self.Layer2 = AttBlock(len_seq=len_seq,
							   input_size=130,
							   output_size=len_seq,
							   hidden_size=hidden_size,
							   num_heads=3,
							   dropout=dropout)

		self.RedLayer = MaxLayer() #MaxLayer()  # MeanLayer()
		self.Dense1   = nn.Sequential(nn.Linear(hidden_size, feat_size), nn.ReLU())
		self.Dense2   = nn.Linear(40, 2)

		self.FeatRed  = AttFeat(feat_size, 40)

	def initHidden(self, batch):
		return (torch.zeros(2,batch, self.hidden_size),
				torch.zeros(2,batch, self.hidden_size)) #,
				#self.Layer2.init_hidden(batch))

	def forward(self, X, Fe, ret_vec=False):
		x1 = self.dro1(X)
		#x1 = X
		# hc1   = self.initHidden(x1.shape[0])
		# x1, _ = self.Layer1(x1, hc1)
		hc2   = self.Layer2.init_hidden(x1.shape[0])
		x1    = self.Layer2(x1, hc2)

		
		x = self.RedLayer(x1)
		y = self.Dense1(x)
		y = self.FeatRed([y, Fe])

		if ret_vec:
			return y
		y1 = self.Dense2(y).squeeze()
		return y1

	def load(self, path):
		self.load_state_dict(torch.load(path))

	def save(self, path):
		torch.save(self.state_dict(), path) 

class LA_Model(nn.Module):
	def __init__(self, hidden_size, len_seq, Embedding, feat_size=61, embeding_shape=(1,300), dropout=0.2):
		super(LA_Model, self).__init__()
		self.criterion1 = nn.CrossEntropyLoss(reduction='none')#weight=torch.Tensor([0.7,0.3]))
		self.criterion2 = nn.CrossEntropyLoss(reduction='none')#weight=torch.Tensor([0.7,0.3]))
		self.hidden_size = hidden_size
		self.feat_size = feat_size

		# word embedding
		self.emb_w   = nn.Embedding.from_pretrained(torch.from_numpy(Embedding)) #, freeze=False)
		
		self.dro1    = nn.Dropout(dropout)
		self.Layer1  = nn.LSTM(input_size= Embedding.shape[1],
						hidden_size= hidden_size,
						batch_first=True,
						bidirectional=True,
						num_layers=2)
		# self.Layer1  = MAN(Embedding.shape[1], hidden_size, memory_size=100)
		self.Layer2 = AttBlock(len_seq=len_seq,
							   input_size= hidden_size*2,
							   output_size=len_seq,
							   hidden_size=hidden_size,
							   num_heads=2,
							   dropout=dropout)

		self.nora     = nn.BatchNorm1d(hidden_size*2)
		self.nora2    = nn.BatchNorm1d(hidden_size)

		self.RedLayer = MaxLayer() #AddNormLayer() #MaxLayer()  # MeanLayer()
		self.Dense1   = nn.Sequential(nn.Linear(hidden_size, feat_size), nn.LeakyReLU())
		self.Task1   = nn.Linear(64, 2)
		self.Task2   = nn.Linear(64, 2)
		self.Dense3   = nn.Sequential(nn.Linear(100, feat_size), nn.LeakyReLU())
		self.Dense4   = nn.Sequential(nn.Linear(768, 192), nn.LeakyReLU(), nn.Linear(192, feat_size), nn.LeakyReLU())

		self.FeatRed  = AttFeat(feat_size, 64)

	def initHidden(self, batch):
		return (torch.zeros(4,batch, self.hidden_size),
				torch.zeros(4,batch, self.hidden_size))

	def forward(self, X, Fe, ret_vec=False, multi=False):
		# Cause IG is pressent, the last 100 fetures are separated, and 768 from bert
		fe1, fe2 = Fe[:, :self.feat_size], self.Dense3(Fe[:, self.feat_size:(self.feat_size+100)])
		fe3 = self.Dense4(Fe[:, (self.feat_size+100):])

		x1    = self.dro1(self.emb_w(X))
		hc1   = self.initHidden(x1.shape[0])
		x1, _ = self.Layer1(x1, hc1)

		x1 = x1.permute((0,2,1))
		x1 = self.nora(x1)
		x1 = x1.permute((0,2,1))

		hc2   = self.Layer2.init_hidden(x1.shape[0])
		x1    = self.Layer2(x1, hc2)

		# x1 = self.Layer1(x1)

		# x1 = x1.permute((0,2,1))
		# x1 = self.nora2(x1)
		# x1 = x1.permute((0,2,1))
		
		x = self.RedLayer(x1)
		y = self.Dense1(x)
		y = self.FeatRed([y, fe1, fe2, fe3])

		if ret_vec:
			return y
		y1 = self.Task1(y).squeeze()
		y2 = self.Task2(y).squeeze()
		
		return y1, y2


	def load(self, path):
		self.load_state_dict(torch.load(path))

	def save(self, path):
		torch.save(self.state_dict(), path) 

class ManNet_S(nn.Module):
	def __init__(self, hidden_size, len_seq, feat_size=49, dropout=0.2):
		super(ManNet_S, self).__init__()
		self.criterion1 = nn.CrossEntropyLoss()
		self.hidden_size = hidden_size
		self.memory_size = 100
		
		self.dro1    = nn.Dropout(dropout)
		self.Layer1  = MAN(130, hidden_size, memory_size=self.memory_size)

		self.RedLayer = MeanLayer()
		self.Dense1   = nn.Sequential(nn.Linear(hidden_size*2, feat_size), nn.ReLU())
		self.Dense2   = nn.Linear(40, 2)

		self.FeatRed  = AttFeat(feat_size, 40)

	def forward(self, X, Fe, ret_vec=False):
		x = self.dro1(X)
		x = self.Layer1(x)

		x = self.RedLayer(x)
		x = self.Dense1(x)
		y = self.FeatRed([x, Fe])
		
		if ret_vec:
			return y
		y = self.Dense2(y)
		return y.squeeze()

	def load(self, path):
		self.load_state_dict(torch.load(path))
		self.Layer1.M = np.fromfile(path+'.M', dtype=np.float32).reshape((self.memory_size,-1))

	def save(self, path):
		torch.save(self.state_dict(), path) 
		self.Layer1.M.tofile(path+'.M')

class SageModel(nn.Module):
	def __init__(self, in_size, layer, num_inputs=2):
		super(SageModel,self).__init__()
		self.criterion1 = nn.CrossEntropyLoss()
		self.num_inputs = num_inputs
		self.Dense = nn.Linear(layer,2)
		self.RED   = AttFeat(in_size//self.num_inputs,layer)

	def forward(self, X, F, ret_vec=False):
		feat = []
		l = X.shape[1] // self.num_inputs
		for i in range(self.num_inputs):
			feat.append(X[:,l*i:l*(i+1)])
		x = self.RED(feat)
		return self.Dense(x).squeeze()

	def load(self, path):
		self.load_state_dict(torch.load(path))
	def save(self, path):
		torch.save(self.state_dict(), path) 

# =========================================================================
def makeModel(neuronas, leng_seq, dropout=0.2, Enb = None, arquitectura=0):
	torch.manual_seed(12345)
	np.random.seed(123)

	feat_size = 61
	max_vec_size = 19
	
	if arquitectura == 'lstm':
		return LstmAtt(neuronas, leng_seq, dropout=dropout,Embedding=Enb, feat_size=feat_size)
	elif arquitectura == 'lstm_s':
		return LstmAtt_S(neuronas, max_vec_size, dropout=dropout, feat_size=feat_size)
	elif arquitectura == 'mann_s':
		return ManNet_S(neuronas, max_vec_size, dropout=dropout, feat_size=feat_size)
	elif arquitectura == 'mann':
		return ManNet(neuronas, leng_seq, dropout=dropout,Embedding=Enb, feat_size=feat_size)
	elif arquitectura == 'conv':
		return ConvNet(neuronas, leng_seq, dropout=dropout,Embedding=Enb, feat_size=feat_size)
	elif arquitectura == 'la':
		return LA_Model(neuronas, leng_seq, dropout=dropout,Embedding=Enb, feat_size=feat_size)


class HaSpeed2Dataset(Dataset):
	def __init__(self, csv_file):
		self.data_frame = pd.read_csv(csv_file)

	def __len__(self):
		return len(self.data_frame)

	def trans(self, text):
		text = [ int(i) for i in text.split() ]
		return torch.Tensor(text).long()

	def transF(self, text):
		text = [ float(i) for i in text.split() ]
		return torch.Tensor(text).float()

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
			
		sent  = self.trans(self.data_frame.iloc[idx, 1])
		value = self.data_frame.iloc[idx, 2]
		feat  = self.transF(self.data_frame.iloc[idx, 3])
		inx   = int(self.data_frame.iloc[idx, 0])

		sample = {'x1': sent, 'y': value , 'f':feat, 'i': inx}
		return sample

class HaSentDataset(Dataset):
	def __init__(self, csv_file):
		self.data_frame = pd.read_csv(csv_file)

	def __len__(self):
		return len(self.data_frame)

	def trans(self, text, vecs=19):
		text = [ float(i) for i in text.split() ]
		return torch.Tensor(text).reshape(vecs,-1)

	def transF(self, text):
		text = [ float(i) for i in text.split() ]
		return torch.Tensor(text)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		inx   = self.data_frame.loc[idx, 'id']
		sent  = self.trans(self.data_frame.loc[idx, 'vectors'])
		value = self.data_frame.loc[idx, 'hs']
		feat  = self.transF(self.data_frame.loc[idx, 'feat'])

		sample = {'x1': sent, 'y': value, 'i':inx, 'f':feat}
		return sample

class SageDataset(Dataset):
	def __init__(self, csv_file):
		self.data_frame = pd.read_csv(csv_file)

	def __len__(self):
		return len(self.data_frame)

	def trans(self, text):
		text = [ float(i) for i in text.split() ]
		return torch.Tensor(text)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		inx   = self.data_frame.loc[idx, 'id']
		sent  = self.trans(self.data_frame.loc[idx, 'vec'])
		feat  = 0
		value = self.data_frame.loc[idx, 'hs']
		
		sample = {'x1': sent, 'y':value, 'i':inx, 'f':feat}
		return sample

def makeData(filepath, batch, second=False, shuffle=True):
	data   =  None
	if not second:
		data = HaSpeed2Dataset(filepath)
	else:
		data = HaSentDataset(filepath)
	loader =  DataLoader(data, batch_size=batch,
							shuffle=shuffle, num_workers=4,
							drop_last=False)
	return data, loader

def makeDataSa(filepath, batch, shuffle=True):
	data   =  SageDataset(filepath)
	loader =  DataLoader(data, batch_size=batch,
							shuffle=shuffle, num_workers=4,
							drop_last=False)
	return data, loader

def runSageTrain(batch):
	LR    = 0.01
	EPOCH = 20
	LAYER = 10
	model = SageModel(80, LAYER, num_inputs=2)
	initModel(model)

	# Training the sage model as Final Layer

	print('# Training Last Model')
	train, train_l = makeDataSa('data/train_sage.csv', batch)
	val, val_l     = makeDataSa('data/dev_sage.csv', batch)

	board = TorchBoard()
	total_acc_tr, total_acc_val = Train(model, train_l, val_l, 
									  lr=LR, epochs=EPOCH, board=board,
									  save_path='models/hate_sage.pt')
	board.show('photos/model.png')
	del board
	del train_l
	del train
	del val_l
	del val

	print ('# END: train %s | val %s' % (total_acc_tr.max(), total_acc_val.max()))

	# Making Prediction if the data exist
	if os.path.isfile('data/test_sage.csv'):
		save_name = 'data/prediction.tsv'
		print ('# Making Predictions as', colorizar(os.path.basename(save_name)))

		test, test_l = makeDataSa('data/test_sage.csv', batch, shuffle=False)
		EvalData(model, test_l, filepath='models/hate_sage.pt',
				 save_path=save_name, 
				 header=('id', 'hs', ''), label_prediction=True)
		del test_l
		del test
	del model
