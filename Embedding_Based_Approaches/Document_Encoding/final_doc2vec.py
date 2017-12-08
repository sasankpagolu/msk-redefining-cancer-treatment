import string
import numpy as np
import pandas as pd
from sklearn import *
from gensim import utils
from sklearn.svm import SVC
from random import shuffle
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from gensim.models.doc2vec import LabeledSentence
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from gensim.models import Doc2Vec

# Loading Dataset

train = pd.read_csv('./training_variants')
test = pd.read_csv('./stage2_test_variants.csv')
trainx = pd.read_csv('./training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('./stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train = pd.merge(train, trainx, how='left', on='ID').fillna('')

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

def cleanup(text):
    text = text.lower()
    # text= text.translate(str.maketrans("","", string.punctuation))
    return text
train['Text'] = train['Text'].apply(cleanup)
test['Text'] = test['Text'].apply(cleanup)
allText = train['Text'].append(test['Text'],ignore_index=True)

def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

sentences = constructLabeledSentences(allText)

## Training DOC2VEC Model on Kaggle Data

# model = Doc2Vec(min_count=1, window=5, size=100, sample=1e-4, negative=5, workers=8,iter=100,seed=1)

# model.build_vocab(sentences)

# print "Training Model"
# model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

# model.save('./docEmbeddings.d2v')

print "Loading Model ..."
model = Doc2Vec.load('./docEmbeddings.d2v')
print "Model Loaded"

train_arrays = np.zeros((train.shape[0], 100))
train_labels = np.zeros(train.shape[0])

# Creating doc2vec representation for training and testig data
for i in range(train.shape[0]):
    train_arrays[i] = model.docvecs['Text_'+str(i)]
    train_labels[i] = train["Class"][i]

test_arrays = np.zeros((test.shape[0], 100))
for i in range(train.shape[0],allText.shape[0]):
    test_arrays[i-train.shape[0]] = model.docvecs['Text_'+str(i)]

y = train_labels - 1 #fix for zero bound array

xgb = XGBClassifier(max_depth=4,
                          objective='multi:softprob',
                          learning_rate=0.03333)
# Fitting the model
xgb.fit(train_arrays, y)
probas = xgb.predict_proba(train_arrays)
print('Log loss: {}'.format(log_loss(y, probas)))
pred_indices = np.argmax(probas, axis=1)
classes = np.unique(y)
preds = classes[pred_indices]
print('Accuracy: {}'.format(accuracy_score(y, preds)))

probas = xgb.predict_proba(test_arrays)
submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
submission_df['ID'] = test['ID']
submission_df.to_csv('submission_xgb.csv', index=False)

