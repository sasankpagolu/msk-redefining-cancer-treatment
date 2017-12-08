''' README
Takes a text file as input, trains a word2vec model on our text vocab 
Vector dimension set to 100
Also used to creates a numpy array of learned Word2vec embeddings for each instance in the text file.
Also creates a dictionary mapping each word in the text file to a 100 dimensional vector embedding.

Steps to perform before running:
1) Make sure you have numpy,pandas,gensim,nltk packages installed for python.
2) Change the model path to save our model in line#27
3) Input training text file path to be set in trainingtextpath variable in line#29
4) Input test text file path to be set in testtextpath variable in line#30
5) To store the vectors mapped to the training text instances in our input file in a dictionary, set the output file 
   path in trainoutput variable in line#31
6)To store the vectors mapped to the test text instances in our input file in a dictionary, set the output file 
   path in testoutput variable in line#32
'''

import pandas as pd
import os
import gensim
import numpy as np
import nltk
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

modelpath='/home/vp12/w2vmodeladdeddata'

trainingtextpath='training_text'
testtextpath='stage2_test_text.csv'
trainoutput='mean_embedded_selftrain_addeddatamodel_onlytrain.npy'
testoutput='mean_embedded_selftrain_test_addeddatamodel.npy'


class MySentences(object):
    def __init__(self, *arrays):
        self.arrays = arrays
 
    def __iter__(self):
        for array in self.arrays:
            for document in array:
                for sent in nltk.sent_tokenize(document):
                    yield nltk.word_tokenize(sent)

def get_word2vec(sentences, location):
    
    if os.path.exists(location):
        print('Found {}'.format(location))
        model = gensim.models.Word2Vec.load(location)
        return model
    
    print('{} not found. training model'.format(location))
    
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    print('Model done training. Saving to disk')
    model.save(location)
    return model

df_train = pd.read_csv(trainingtextpath, sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_train.head()

df_test = pd.read_csv(testtextpath, sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_test.head()

#df_add = pd.read_csv('test_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
#df_add.head()

#df_train_latest = pd.read_csv('training_text_latest', sep=',', header=None, skiprows=1, names=["ID","Text"])
#df_train_latest.head()


w2vec = get_word2vec(
    MySentences(
        df_train['Text'].values.astype('U'), 
        df_test['Text'].values.astype('U'),  #Commented for Kaggle limits
        #df_add['Text'].values.astype('U'),
    ),
    modelpath
)

class MyTokenizer:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_X = []
        for document in X:
            tokenized_doc = []
            for sent in nltk.sent_tokenize(document):
                tokenized_doc += nltk.word_tokenize(sent)
            transformed_X.append(np.array(tokenized_doc))
        return np.array(transformed_X)
    
    def fit_transform(self, X, y=None):
        return self.transform(X)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.wv.syn0[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = MyTokenizer().fit_transform(X)
        
        return np.array([
            np.mean([self.word2vec.wv[w] for w in words if w in self.word2vec.wv]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2vec)
mean_embedded = mean_embedding_vectorizer.fit_transform(df_train['Text'].values.astype('U'))
#mean_embedded_latest = mean_embedding_vectorizer.fit_transform(df_train_latest['Text'].values.astype('U'))
mean_embedded_test = mean_embedding_vectorizer.transform(df_test['Text'].values.astype('U'))
np.save(trainoutput,mean_embedded)
np.save(testoutput,mean_embedded_test)