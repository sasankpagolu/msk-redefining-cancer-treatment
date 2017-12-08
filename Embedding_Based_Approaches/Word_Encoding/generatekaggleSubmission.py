'''
README: Generates the kaggle submission files of the model on test files.

install all the dependencies
Put your train embeddings path and test embeddings path in varaibles in line# 27 and 28
Put all your training files and test files in the same folder as this code.
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

# Train labels and glove,word2vec embeddings

embeddingstrain='mean_embedded_selftrain_addeddatamodel_latest.npy'
embeddingstest='mean_embedded_selftrain_test_addeddatamodel.npy'
X_train_wvec=np.load(embeddingstrain)
X_test_wvec=np.load(embeddingstest)
y_train=[]
fp=open('training_variants')
next(fp)
for line in fp:
    values=line.split(',')
    y_train.append(int(values[3].strip('\n')))

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

df_test_txt = pd.read_csv('stage2_test_text.csv', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_test_txt.head()
df_test_var = pd.read_csv('stage2_test_variants.csv')
df_test_var.head()
df_test = pd.merge(df_test_var, df_test_txt, how='left', on='ID')
df_test.head()

gene_le = LabelEncoder()
gene_encoded_test = gene_le.fit_transform(df_test['Gene'].values.ravel()).reshape(-1, 1)
gene_encoded_test = gene_encoded_test / np.max(gene_encoded_test)

variation_le = LabelEncoder()
variation_encoded_test = variation_le.fit_transform(df_test['Variation'].values.ravel()).reshape(-1, 1)
variation_encoded_test = variation_encoded_test / np.max(variation_encoded_test)

algo = XGBClassifier(max_depth=8,objective='multi:softprob',learning_rate=0.03333)
#np.hstack((gene_encoded, variation_encoded,X_train_wvec))
algo.fit(np.hstack((gene_encoded, variation_encoded,X_train_wvec)), y_train)
#np.hstack((gene_encoded_test, variation_encoded_test,X_test_wvec))
probas = algo.predict_proba(np.hstack((gene_encoded_test, variation_encoded_test,X_test_wvec)))
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.to_csv('ns_XGB_enc_latest.csv', index=False)


algo = RandomForestClassifier(n_estimators=2000)
#np.hstack((gene_encoded, variation_encoded,X_train_wvec))
algo.fit(np.hstack((gene_encoded, variation_encoded,X_train_wvec)), y_train)
#(gene_encoded_test, variation_encoded_test,X_test_wvec))
probas = algo.predict_proba(np.hstack((gene_encoded_test, variation_encoded_test,X_test_wvec)))
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.to_csv('ns_rf_enc_latest.csv', index=False)

gbm = lgb.LGBMClassifier(objective='multiclass',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(np.hstack((gene_encoded, variation_encoded,X_train_wvec)), y_train)
probas = gbm.predict_proba(np.hstack((gene_encoded_test, variation_encoded_test,X_test_wvec)))
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()
submission_df.to_csv('ns_gbm_enc_latest.csv', index=False)


'''    
X_train_glove=np.load('output_glove_train.npy')
X_train_wvec=np.load('output.npy')

#print values[1]

evaluate_features(X,y,RandomForestClassifier(n_estimators=1000))
evaluate_features(X,y,SVC(kernel='linear',probability=True))

from sklearn.datasets import load_iris
evaluate_features(*load_iris(True))


#Test Labels and glove,word2vec Embeddings

X_test_glove=np.load('output_test_glove.npy')
X_test_wvec=np.load('output_test_wvec.npy')
'''



'''
mlp_w2vec = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 2), random_state=1)
mlp_w2vec.fit(X_train_wvec, y_train)
probas = mlp_w2vec.predict_proba(X_test_wvec)
submission_df['ID'] = df_test['ID']
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df.to_csv('submission_mlp_wvec.csv', index=False)

mlp_gl = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 2), random_state=1)
mlp_gl.fit(X_train_glove, y_train)
probas = mlp_gl.predict_proba(X_test_glove)
submission_df['ID'] = df_test['ID']
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df.to_csv('submission_mlp_gl.csv', index=False)


xgb_w2vec = XGBClassifier(max_depth=4,
                          objective='multi:softprob',
                          learning_rate=0.03333)
xgb_w2vec.fit(X_train_wvec, y_train)
probas = xgb_w2vec.predict_proba(X_test_wvec)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df.insert(0, 'ID', df_test['ID'])
#submission_df['ID'] = df_test['ID']
submission_df.to_csv('submission_Xgb_wvec.csv', index=False)

xgb_gl = XGBClassifier(max_depth=4,
                          objective='multi:softprob',
                          learning_rate=0.03333)
xgb_gl.fit(X_train_glove, y_train)
probas = xgb_gl.predict_proba(X_test_glove)

#submission_df['ID'] = df_test['ID']
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df.insert(0, 'ID', df_test['ID'])
submission_df.to_csv('submission_Xgb_gl.csv', index=False)


# Evaluate Wvec
lr_wvec = LogisticRegression()
lr_wvec.fit(X_train_wvec, y_train)

probas = lr_wvec.predict_proba(X_test_wvec)
submission_df['ID'] = df_test['ID']
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df.head()
submission_df.to_csv('submission_wvec_lr.csv', index=False)


# Evaluate glove
lr_gl = LogisticRegression()
lr_gl.fit(X_train_glove, y_train)

probas = lr_gl.predict_proba(X_test_glove)
submission_df['ID'] = df_test['ID']
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df.head()
submission_df.to_csv('submission_glove_lr.csv', index=False)

# Evaluate Wvec
rf_wvec = RandomForestClassifier(n_estimators=1000)
rf_wvec.fit(X_train_wvec, y_train)

probas = rf_wvec.predict_proba(X_test_wvec)

submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()

submission_df.to_csv('submission_wvec_rf.csv', index=False)


# Evaluate glove
rf_gl = RandomForestClassifier(n_estimators=1000)
rf_gl.fit(X_train_glove, y_train)

probas = rf_gl.predict_proba(X_test_glove)

submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()

submission_df.to_csv('submission_glove_rf.csv', index=False)

# Evaluate Wvec
rf_wvec = SVC(kernel='linear',probability=True)
rf_wvec.fit(X_train_wvec, y_train)

probas = rf_wvec.predict_proba(X_test_wvec)

submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()

submission_df.to_csv('submission_wvec_svm_linear.csv', index=False)


# Evaluate glove
rf_gl = SVC(kernel='linear',probability=True)
rf_gl.fit(X_train_glove, y_train)

probas = rf_gl.predict_proba(X_test_glove)

submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()

submission_df.to_csv('submission_glove_svm_linear.csv', index=False)

# Evaluate Wvec
rf_wvec = SVC(kernel='ploy',probability=True)
rf_wvec.fit(X_train_wvec, y_train)

probas = rf_wvec.predict_proba(X_test_wvec)

submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()

submission_df.to_csv('submission_wvec_svm_poly.csv', index=False)


# Evaluate glove
rf_gl = SVC(kernel='poly',probability=True)
rf_gl.fit(X_train_glove, y_train)

probas = rf_gl.predict_proba(X_test_glove)

submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()

submission_df.to_csv('submission_glove_svm_poly.csv', index=False)


'''


