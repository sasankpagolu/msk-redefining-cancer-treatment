#New Approach

import pandas as pd
import numpy as np
import lightgbm as lgb
# uncomment for Pyton 2.7
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
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

import gensim

import nltk

from xgboost import XGBClassifier

import os

from nltk.tokenize import word_tokenize

def word_count(str):
    str = unicode(str,'utf8')
    x = word_tokenize(str)
    return len(x)
def char_count(str):
	return len(str)

def count(str,a):
    str = unicode(str,'utf8')
    #a = unicode(a,'utf8')
    x = word_tokenize(str)
    i=0
    for y in x:
        if y.lower()==a.lower():
            i+=1;

    return i


# Read training file
df_train_txt = pd.read_csv('training_text',sep='\|\|',encoding = 'utf8', header=None, skiprows=1, names=["ID","Text"])
df_train_txt.head()

df_train_var = pd.read_csv('training_variants',encoding = 'utf8')
df_train_var.head()

df_test_txt = pd.read_csv('stage2_test_text.csv',encoding = 'utf8', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_test_txt.head()

df_test_var = pd.read_csv('stage2_test_variants.csv',encoding = 'utf8')
df_test_var.head()

df_train = pd.merge(df_train_var, df_train_txt, how='left', on='ID')
df_train.head()

df_test = pd.merge(df_test_var, df_test_txt, how='left', on='ID')
df_test.head()
print("reading done")
print(df_train_txt.Text[1])



len_words_train = list()
len_chars_train = list()
len_genes_train = list()


len_words_test = list()
len_chars_test = list()
len_genes_test = list()

gene_len_train = list()
gene_char_count_train = list()

variant_len_train = list()
variant_char_count_train = list()

gene_len_test = list()
gene_char_count_test = list()

variant_len_test = list()
variant_char_count_test = list()

for idx in range(df_train.Text.shape[0]):
    text = (df_train.Text[idx])
    text  =str(text)
    len_words_train.append(word_count(text))
    len_chars_train.append(char_count(text))
    len_genes_train.append(count(text,df_train.Gene[idx]))
    gene = df_train.Gene[idx]
    gene = str(gene)
    gene_len_train.append(word_count(gene))
    gene_char_count_train.append(char_count(gene))
    variant_len_train.append(word_count(str(df_train.Variation[idx])))
    variant_char_count_train.append(char_count(str(df_train.Variation[idx])))


for idx in range(df_test.Text.shape[0]):
    text = (df_test.Text[idx])
    text  =str(text)
    len_words_test.append(word_count(text))
    len_chars_test.append(char_count(text))
    len_genes_test.append(count(text,df_test.Gene[idx]))
    gene = df_test.Gene[idx]
    gene = str(gene)
    gene_len_test.append(word_count(gene))
    gene_char_count_test.append(char_count(gene))
    variant_len_test.append(word_count(str(df_test.Variation[idx])))
    variant_char_count_test.append(char_count(str(df_test.Variation[idx])))

print('Step1 Done')

count_vectorizer_text = TfidfVectorizer( strip_accents ='ascii',
    analyzer="word", tokenizer=nltk.word_tokenize,stop_words='english',
    max_features=None,ngram_range=(1,2))

count_vectorizer_gene = TfidfVectorizer( strip_accents ='ascii',
    analyzer="char", tokenizer=nltk.word_tokenize,stop_words='english',
    max_features=None,ngram_range=(1,10))
    
count_vectorizer_var = TfidfVectorizer( strip_accents ='ascii',
    analyzer="char", tokenizer=nltk.word_tokenize,stop_words='english',
    max_features=None,ngram_range=(1,10))


tfidf_train_text = count_vectorizer_text.fit_transform(df_train['Text'].values.astype('U'))

tfidf_test_text = count_vectorizer_text.transform(df_test['Text'].values.astype('U'))

tfidf_train_gene = count_vectorizer_gene.fit_transform(df_train['Gene'].values.astype('U'))

tfidf_test_gene = count_vectorizer_gene.transform(df_test['Gene'].values.astype('U'))

tfidf_train_var = count_vectorizer_var.fit_transform(df_train['Variation'].values.astype('U'))

tfidf_test_var = count_vectorizer_var.transform(df_test['Variation'].values.astype('U'))

print('Step::2 Done')

len_genes_train = np.array(len_genes_train)
len_words_train = np.array(len_words_train)
len_chars_train	= np.array(len_chars_train)
len_genes_train = np.reshape(len_genes_train,[len(len_genes_train),1])
len_words_train = np.reshape(len_words_train,[len(len_words_train),1])
len_chars_train = np.reshape(len_chars_train,[len(len_chars_train),1])

gene_len_train = np.array(gene_len_train)
gene_len_train = np.reshape(gene_len_train,[len(gene_len_train),1])

gene_char_count_train = np.array(gene_char_count_train)
gene_char_count_train = np.reshape(gene_char_count_train,[len(gene_char_count_train),1])

variant_len_train = np.array(variant_len_train)
variant_len_train = np.reshape(variant_len_train,[len(variant_len_train),1])

variant_char_count_train = np.array(variant_char_count_train)
variant_char_count_train = np.reshape(variant_char_count_train,[len(variant_char_count_train),1])


len_genes_test = np.array(len_genes_test)
len_words_test = np.array(len_words_test)
len_chars_test	= np.array(len_chars_test)

len_genes_test = np.reshape(len_genes_test,[len(len_genes_test),1])
len_words_test = np.reshape(len_words_test,[len(len_words_test),1])
len_chars_test = np.reshape(len_chars_test,[len(len_chars_test),1])

gene_len_test = np.array(gene_len_test)
gene_len_test = np.reshape(gene_len_test,[len(gene_len_test),1])

gene_char_count_test = np.array(gene_char_count_test)
gene_char_count_test = np.reshape(gene_char_count_test,[len(gene_char_count_test),1])

variant_len_test = np.array(variant_len_test)
variant_len_test = np.reshape(variant_len_test,[len(variant_len_test),1])

variant_char_count_test = np.array(variant_char_count_test)
variant_char_count_test = np.reshape(variant_char_count_test,[len(variant_char_count_test),1])

# len_genes = coo_matrix(len_genes)
# len_chars = coo_matrix(len_chars)
# len_words = coo_matrix(len_words)


svd_text = TruncatedSVD(n_components= 50)
truncated_tfidf_train_text = svd_text.fit_transform(tfidf_train_text)
truncated_tfidf_test_text = svd_text.transform(tfidf_test_text)

svd_var = TruncatedSVD(n_components= 20)
truncated_tfidf_train_var = svd_var.fit_transform(tfidf_train_var)
truncated_tfidf_test_var = svd_var.transform(tfidf_test_var)

svd_gene = TruncatedSVD(n_components= 20)
truncated_tfidf_train_gene = svd_gene.fit_transform(tfidf_train_gene)
truncated_tfidf_test_gene = svd_gene.transform(tfidf_test_gene)
# truncated_tfidf_train = (np.array(tfidf_train.todense()))
# truncated_tfidf_test = (np.array(tfidf_test.todense()))

gene_le = LabelEncoder()
gene_encoded_train = gene_le.fit_transform(df_train['Gene'].values.ravel()).reshape(-1, 1)
gene_encoded_train = gene_encoded_train / np.max(gene_encoded_train)

gene_encoded_test = gene_le.fit_transform(df_test['Gene'].values.ravel()).reshape(-1, 1)
gene_encoded_test = gene_encoded_test / np.max(gene_encoded_test)

variation_le = LabelEncoder()
variation_encoded_train = variation_le.fit_transform(df_train['Variation'].values.ravel()).reshape(-1, 1)
variation_encoded_train = variation_encoded_train / np.max(variation_encoded_train)

variation_encoded_test = variation_le.fit_transform(df_test['Variation'].values.ravel()).reshape(-1, 1)
variation_encoded_test = variation_encoded_test / np.max(variation_encoded_test)


features_train = np.hstack([truncated_tfidf_train_text,truncated_tfidf_train_gene,truncated_tfidf_train_var,len_genes_train,len_chars_train,len_words_train,gene_encoded_train,variation_encoded_train])
features_train = np.hstack([features_train,gene_len_train,gene_char_count_train,variant_char_count_train,variant_len_train])
features_test = np.hstack([truncated_tfidf_test_text,truncated_tfidf_test_gene,truncated_tfidf_test_var,len_genes_test,len_chars_test,len_words_test,gene_encoded_test,variation_encoded_test])
features_test = np.hstack([features_test,gene_len_test,gene_char_count_test,variant_char_count_test,variant_len_test])
np.save('features_train.npy',features_train)
np.save('features_test.npy',features_test)

# df_test_txt = pd.read_csv('stage2_test_text.csv', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
# df_test_txt.head()
# df_test_var = pd.read_csv('stage2_test_variants.csv')
# df_test_var.head()
# df_test = pd.merge(df_test_var, df_test_txt, how='left', on='ID')
# df_test.head()

#---------#

clf = LogisticRegression()
X = features_train
y = df_train['Class']
np.save('train_outputs.npy',y)
print(X.shape)
clf.fit(X,y)
probas = clf.predict_proba((features_test))

submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.to_csv('submission_lr.csv', index=False)

#-----------#
xgb = XGBClassifier(max_depth=4,
                          objective='multi:softprob',
                          learning_rate=0.03333)
xgb.fit(X, y)
probas = xgb.predict_proba(features_test)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.to_csv('submission_xgb.csv', index=False)

#----------------#
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X, y)
probas = rf.predict_proba(features_test)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()
submission_df.to_csv('submission_rf.csv', index=False)

#---------#

print(X.shape)
gbm = lgb.LGBMClassifier(objective='multiclass',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X, y)
probas = gbm.predict_proba(features_test)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = df_test['ID']
submission_df.head()
submission_df.to_csv('submission_gbm.csv', index=False)

# probas_train = gbm.predict_proba(features_train)
# submission_df = pd.DataFrame(probas_train, columns=['class'+str(c+1) for c in range(9)])
# submission_df['ID'] = df_test['ID']
# submission_df.head()
# submission_df.to_csv('submission_gbm_train.csv', index=False)
# pred_indices = np.argmax(probas_train, axis=1)
# classes = np.unique(y)
# preds = classes[pred_indices]

# print('Log loss: {}'.format(log_loss(y, probas_train)))
# print('Accuracy: {}'.format(accuracy_score(y, preds)))
