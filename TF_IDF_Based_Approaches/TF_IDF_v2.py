import os
import re
import nltk
import pickle
import string
import gensim
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from nltk.corpus import stopwords
import scikitplot.plotters as skplt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def evaluate_features(X, y, clf):

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)
    clf.fit(train_x, train_y)
    probas = clf.predict_proba(test_x)
    print('Log loss: {}'.format(log_loss(test_y, probas)))
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(test_y)
    preds = classes[pred_indices]
    print('Accuracy: {}'.format(accuracy_score(test_y, preds)))

# Loading Datasets

train_variant = pd.read_csv("./training_variants")
test_variant = pd.read_csv("./stage2_test_variants.csv")

train_text = pd.read_csv("./training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("./stage2_test_text.csv", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

# Creating Merged Dataset
train = pd.merge(train_variant, train_text, how='left', on='ID')
test = pd.merge(test_variant, test_text, how='left', on='ID')

train_y = train['Class'].values
train_x = train.drop('Class', axis=1)

# Creating TFIDF Vector after tokenizing words and removing stopwords
tfidf_vectorizer = TfidfVectorizer(analyzer="word", tokenizer=nltk.word_tokenize,preprocessor=None, stop_words='english', max_features=None)    

# Extracting TFIDF features from training data
tfidf_train = tfidf_vectorizer.fit_transform(train_x['Text'])

# Applying SVD on the extracted features
svd = TruncatedSVD(n_components=25, n_iter=25, random_state=12)
truncated_tfidf = svd.fit_transform(tfidf_train)


print "Training data evaluation"
print "RDF truncated"
evaluate_features(truncated_tfidf, train_y.ravel(), 
                  RandomForestClassifier(n_estimators=1000, max_depth=5, verbose=1))

print "SVC"
evaluate_features(tfidf_train, train_y.ravel(), 
                  SVC(kernel='linear', probability=True))

print "Logistic Regression"
evaluate_features(tfidf_train, train_y.ravel())

print "RDF tfidf"
evaluate_features(tfidf_train, train_y.ravel(),
                  RandomForestClassifier(n_estimators=1000, max_depth=15, verbose=1))

print "Testing data evaluation"
tfidf_test = tfidf_vectorizer.transform(test['Text'])

print "SVC"
svc = SVC(kernel='linear', probability=True)
svc.fit(tfidf_train, train['Class'])
probas = svc.predict_proba(tfidf_test)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = test['ID']
submission_df.to_csv('submission_1.csv', index=False)

print "RDF tfidf"
rdf = RandomForestClassifier(n_estimators=1000, max_depth=15, verbose=1)
rdf.fit(tfidf_train, train['Class'])
probas = rdf.predict_proba(tfidf_test)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
print test['ID']
submission_df['ID'] = test['ID']
submission_df.to_csv('submission_2.csv', index=False)
