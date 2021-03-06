# Italian Deep Ensemble Model 
This is the model used on Task A: Hate Speech Detection at Evalita 2020.

## Requirements
First clone or download the project as a zip file. Then on the location of the downloaded file(or clonned folder):
```bach 
cd evalita_20_hate
pip install -r requirements.txt
```

## Usage

This project uses [FreeLing 4.0](http://nlp.lsi.upc.edu/freeling/index.php/node/30) Tool. To archive a successful performance, you most install FreeLing with the python API and run:
```bash
cp $FREELINGDIR/share/freeling/APIs/python3/_pyfreeling.so  code/_pyfreeling.so
cp $FREELINGDIR/share/freeling/APIs/python3/pyfreeling.py  code/pyfreeling.py
```

where $FREELINGDIR is the installation path of Freeling (deafult: /usr/local)

The data is placed inside a data folder:
```bash
mkdir data
cp DIRDATA/haspeede2_dev_taskAB.tsv data/haspeede2_dev_taskAB.tsv
cp DIRDATA/haspeede2-test_taskAB-tweet.tsv data/haspeede2-test_taskAB-tweet.tsv
cp DIRDATA/haspeede2-test_taskAB-news.tsv data/haspeede2-test_taskAB-news.tsv
cp DIRDATAOLD/training_set_senticpolc16.tsv data/training_set_senticpolc16.tsv
```
where DIRDATA is the data location for haspeede2 dataset and DIRDATAOLD for sentipolc_2016 dataset.

The Pre-Trained [BERT](https://huggingface.co/dbmdz/bert-base-italian-cased) model from transformer python library was used. The path to this models lies in code/ig_frature.py file with the variable name BERT_PATH:
```python
# pre-trained Bert folder path
BERT_PATH = '/DATA/Mainstorage/Prog/NLP/dbmdz/bert-base-italian-uncased'
bert_model = None
bert_tk = None
```

To predict from unlabeled data, first change the test data inside main.py:
```python
DATA_PRED_PATH = 'data/haspeede2_test_taskAB-tweets.tsv'
# DATA_PRED_PATH = 'data/haspeede2-test_taskAB-news.tsv'
```
comment the one that is not relevant an finally to train the model run:
```bash
python main.py
```

## Hyper-parameters
The hyperparameters can be changed inside the main.py code, but also though cosole. To see the whole list run:
```bach
python main.py --help
```

## Dataset urls:
* Evalita_2020: http://www.di.unito.it/~tutreeb/haspeede-evalita20/index.html 
* Senticpolc_2016: http://www.di.unito.it/~tutreeb/sentipolc-evalita16/index.html
