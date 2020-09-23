import os
import re
import pandas as pd
from .preprocesing import sent_prep
from .utils import *

# Opcional
def make(dir_name, corpus_name='_corpus.txt', ret_path=False):
	cp_name = os.path.join(dir_name, corpus_name)
	f = open (cp_name, 'w', encoding='utf-8')
	print ('# Making', colorizar(os.path.basename(cp_name)))

	for file in os.listdir(dir_name):
		if file[-3:] != 'tsv':
			continue

		# solo con tags or lemmas
		if len(re.findall('shuffled',file)) < 1:
			continue

		data_csv = pd.read_csv( os.path.join(dir_name, file), sep='\t')
		bar = StatusBar(len(data_csv))
		for i in range( len(data_csv) ):
			tweet = str ( data_csv.iloc[i, 1] )
			tweets = tweet.replace('\t', ' ').splitlines()
			for tui in tweets:
				ora = sent_prep(tui + ' .', ret_list=True)
				for s in ora:
					f.write(s+'\n')
			bar.update()

		del bar
		del data_csv
	f.close()
	if ret_path:
		return cp_name

if __name__ == '__main__':
	d = '/DATA/work_space/6-IA-ML-DL-RL/2-EvalIta2020/data'
	make(d)

