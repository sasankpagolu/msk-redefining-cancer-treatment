'''
README: Evaluates the cross validation performance of the model on train files. Prints accuracy and Log Loss

install all the dependencies
Put your embeddings path in embeddings varaible in line# 27
Put all your training files in the same folder as this code.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb



embeddings='output_glove_train.npy'

df_train_txt = pd.read_csv('training_text', sep=',', header=None, skiprows=1, names=["ID","Text"])
df_train_txt.head()
df_train_var = pd.read_csv('training_variants')
df_train_var.head()
df_train = pd.merge(df_train_var, df_train_txt, how='left', on='ID')
df_train.head()


gene_le = LabelEncoder()
gene_encoded = gene_le.fit_transform(df_train['Gene'].values.ravel()).reshape(-1, 1)
gene_encoded = gene_encoded / np.max(gene_encoded)

variation_le = LabelEncoder()
variation_encoded = variation_le.fit_transform(df_train['Variation'].values.ravel()).reshape(-1, 1)
variation_encoded = variation_encoded / np.max(variation_encoded)

X_latest=np.load(embeddings)
X=np.hstack((gene_encoded,variation_encoded,X_latest))
y_train_latest=[]
fp=open('training_variants')
next(fp)
for line in fp:
    values=line.split(',')
    y_train_latest.append(int(values[3].strip('\n')))


data=X
labels=y_train_latest
labels_test=labels
algo = lgb.LGBMClassifier(objective='multiclass',num_leaves=31,learning_rate=0.05,n_estimators=20)
probas = cross_val_predict(algo, data, labels, cv=StratifiedKFold(n_splits=5), n_jobs=-1, method='predict_proba')
pred_indices = np.argmax(probas, axis=1)
classes = np.unique(labels_test)
preds = classes[pred_indices]
print('Light GBM on glove')
print('Log loss: {}'.format(log_loss(labels_test, probas)))
print('Accuracy: {}'.format(accuracy_score(labels_test, preds)))
print '\n\n'

algo = XGBClassifier(max_depth=8,objective='multi:softprob',learning_rate=0.03333)
probas = cross_val_predict(algo, data, labels, cv=StratifiedKFold(n_splits=5), n_jobs=-1, method='predict_proba')
pred_indices = np.argmax(probas, axis=1)
classes = np.unique(labels_test)
preds = classes[pred_indices]
print('XGB on glove')
print('Log loss: {}'.format(log_loss(labels_test, probas)))
print('Accuracy: {}'.format(accuracy_score(labels_test, preds)))
print '\n\n'

algo = RandomForestClassifier(n_estimators=2000)
probas = cross_val_predict(algo, data, labels, cv=StratifiedKFold(n_splits=5), n_jobs=-1, method='predict_proba')
pred_indices = np.argmax(probas, axis=1)
classes = np.unique(labels_test)
preds = classes[pred_indices]
print('RF on glove')
print('Log loss: {}'.format(log_loss(labels_test, probas)))
print('Accuracy: {}'.format(accuracy_score(labels_test, preds)))
print '\n\n'