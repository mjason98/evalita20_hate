# In[1]:
from sklearn.feature_extraction.text import TfidfVectorizer as m
from sklearn.feature_selection import mutual_info_classif

import sklearn.svm as t
import pandas as pd

DATA_PATH      = 'data/EvalIta-HateSpeech2018/FB-folder-20200907T154715Z-001/FB-folder/FB-train/haspeede_FB-train.tsv'
DATA_PRED_PATH = 'data/EvalIta-HateSpeech2018/FB-folder-20200907T154715Z-001/FB-folder/FB-reference/haspeede_FB-reference.tsv'

# In[2]:
data = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['id', 'txt', 'y'])
#data.head()

X = data.drop(['id', 'y'], axis=1)
Y = data.drop(['id', 'txt'], axis=1)
X = X.to_numpy()
Y = Y.to_numpy().reshape(-1)

# In[3]:
vectorize = m()
model     = t.LinearSVC()

X = vectorize.fit_transform(X.reshape(-1).tolist())
# In[4]:
model.fit(X, Y)
# In[5]:
data2 = pd.read_csv(DATA_PRED_PATH, sep='\t', header=None, names=['id', 'txt', 'y'])

x_test = data2.drop(['id', 'y'], axis=1).to_numpy()
x_test = vectorize.transform(x_test.reshape(-1).tolist())
# In[6]:
y_pred = model.predict(x_test)
y      = data2.drop(['id', 'txt'], axis=1).to_numpy().reshape(-1)
# In[7]:
acc = (y_pred == y).sum() / y_pred.shape[0]
print(acc)

print(X.shape, x_test.shape)

# In[8]:
# a = pd.Series(list(y_pred))
# b = data2.drop(['y'], axis=1)
# b = pd.concat([b, a], axis=1)
# b.to_csv('baseline.tsv', sep='\t', header=('id', 'txt', 'y'), index=False)

value = mutual_info_classif(X,Y, discrete_features=True)
names = vectorize.get_feature_names()
best  = [(a,b) for a,b in zip(value, names)]
best.sort()
best = best[-50:]

# In[9]

print(best)

# %%
