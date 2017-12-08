# Import Stattements
import numpy as np
import html
import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from nltk import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize.casual import TweetTokenizer
import codecs

#import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input
from keras.layers.wrappers import Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adagrad
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import load_model

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Lambda
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

#import of scipy

from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.ensemble import AdaBoostRegressor
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

import pandas as pd

class Text(object):
	
	def __init__(self,id,text,gene,mutation,category):
		self.id = id
		self.text = text
		self.gene = gene
		self.mutation = mutation
		self.category = category

def read_training_data (training_text,training_variants):
	
	list_train = list();
	k=0
	with codecs.open (training_text,'r','utf8') as file_input:
		for line in file_input:
			line = line.strip()
			array_splits = line.split('||')
			k = k+1
			if k==1:
				print(array_splits)
			if len(array_splits)<2 :
				continue
			list_train.append(Text(array_splits[0],(array_splits[1])," "," "," "))

	i = 0
	k = 0
	print(len(list_train))
	with codecs.open(training_variants,'r','utf8') as file_input:
		for line in file_input:
			line = line.strip()
			k = k+1
			if (k==1):
				continue
			array_splits = line.split(',')
			if len(array_splits)<4 :
				continue
			list_train[i].gene = array_splits[1]
			list_train[i].mutation = array_splits[2]
			list_train[i].category = int(array_splits[3])-1
			i = i+1

	return list_train

def prediction_on_test_hierarchial(training_list,test_list):

	df_test_txt = pd.read_csv('stage2_test_text.csv', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
	df_test_txt.head()
	df_test_var = pd.read_csv('stage2_test_variants.csv')
	df_test_var.head()
	df_test = pd.merge(df_test_var, df_test_txt, how='left', on='ID')
	df_test.head()

	from nltk import tokenize
	
	train_documents = []
	train_labels = []
	train_texts = []

	test_documents = []
	test_texts = []
	
	total_documents = []
	total_texts = []

	for idx in range(len(training_list)):
	    text = (training_list[idx].text)
	    train_texts.append(text)
	    total_texts.append(text)

	    sentences = tokenize.sent_tokenize(text)
	    
	    train_documents.append(sentences)
	    train_labels.append(int(training_list[idx].category))
	    total_documents.append(sentences)

	for idx in range(len(test_list)):
	    text = (test_list[idx].text)
	    test_texts.append(text)
	    total_texts.append(text)

	    sentences = tokenize.sent_tokenize(text)
	    
	    test_documents.append(sentences)
	    total_documents.append(sentences)
	
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(total_texts)

	MAX_SENTS = 100
	MAX_SENT_LENGTH = 20
	data = np.zeros((len(test_texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
	word_index = tokenizer.word_index
	for i, sentences in enumerate(test_documents):
	    for j, sent in enumerate(sentences):
	        if j< MAX_SENTS:
	            wordTokens = text_to_word_sequence(sent)
	            k=0
	            for _, word in enumerate(wordTokens):
	                if k<MAX_SENT_LENGTH:
	                    data[i,j,k] = tokenizer.word_index[word]
	                    k=k+1                    
	                    
	
	vocab_size = len(word_index)
	#EMBEDDING_DIM = 400
# 	glove_train = np.load('./Data/embeddingmap.npy',encoding = 'latin1');
# 	glove_test = np.load('./Data/embeddingmap_test_wvec.npy',encoding = 'latin1')
# 	#glove_train = np.load('embeddingmap.npy',encoding = 'latin1')
# 	#glove_test = np.load('embeddingmap.npy',encoding = 'latin1')
# 	print(len(glove_train.item()))
# 	print(len(glove_test.item()))

# 	embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))
# 	number_found =0
# 	number_not_found = 0
# 	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
# #	print(glove_train.item().keys())
# 	for word, i in word_index.items():
# 	    embedding_vector = glove_train.item().get(word)
# 	    if embedding_vector is not None:
# 	        # words not found in embedding index will be all-zeros.
# 	        embedding_matrix[i] = embedding_vector
# 	        number_found +=1
# #		print('matched')
# #		print(word)
# 	        continue
# 	        #print (hi)	
# 	    embedding_vector = glove_test.item().get(word)
# 	    if embedding_vector is not None:
# 	        # words not found in embedding index will be all-zeros.
# 	        embedding_matrix[i] = embedding_vector
# 	        number_found +=1
# 	        continue
# 	    number_not_found+=1
# #	    print('no match')
# #	    print(word)

# 	print(number_found)
# 	print(number_not_found)
	model = load_model('./models/model_30_hierarchial.h5')
	probas = model.predict(data)
	submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
	submission_df['ID'] = df_test['ID']
	submission_df.to_csv('submission_lstm_hierarchial_30.csv', index=False)

def prediction_on_test(training_list,test_list):

	df_test_txt = pd.read_csv('stage2_test_text.csv', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
	df_test_txt.head()
	df_test_var = pd.read_csv('stage2_test_variants.csv')
	df_test_var.head()
	df_test = pd.merge(df_test_var, df_test_txt, how='left', on='ID')
	df_test.head()

	docs_train = list()
	score_train = list()
	total_dataset = list()
	for text in training_list:
		docs_train.append(text.text);
		score_train.append(int(text.category));
		total_dataset.append(text.text)
	
	#print(total_dataset)
	score_train = to_categorical(score_train, num_classes=9)
	docs_test = list();
	for text in test_list:
		print(text.text)
		docs_test.append(text.text);
		print(text.category)
		#score_test.append(int(text.category));
		total_dataset.append(text.text)

	#print(total_dataset)
	#score_test = to_categorical(score_test, num_classes=9)
	t = Tokenizer(lower=False,filters='\t\n')
	t.fit_on_texts(total_dataset)
	word_index = t.word_index
	print(t.document_count)
	vocab_size = len(t.word_counts)
	print(vocab_size)
	print (len(word_index))
	max_len = 300
	sequences_train = t.texts_to_sequences(docs_train)
	sequences_test = t.texts_to_sequences(docs_test)
	padded_train 	= pad_sequences(sequences_train,maxlen = max_len,padding = 'post')
	padded_test		= pad_sequences(sequences_test,maxlen = max_len,padding = 'post')
	sequences_train = t.texts_to_sequences(docs_train)
	sequences_test = t.texts_to_sequences(docs_test)
	model = load_model('./models/model_20.h5')
	probas = model.predict(padded_test)
	submission_df = pd.DataFrame(probas, columns=['class'+str(c+1) for c in range(9)])
	submission_df['ID'] = df_test['ID']
	submission_df.to_csv('submission_lstm_20.csv', index=False)



training_variants_filename = "training_variants"
training_text_filename = "training_text"

test_filename = "stage2_test_text.csv"
test_variant = "stage2_test_variants.csv"

training_list = read_training_data(training_text_filename,training_variants_filename)
test_list = read_training_data(test_filename,test_variant)
print ("Done Reading File")
prediction_on_test_hierarchial(training_list,test_list)
prediction_on_test(training_list,test_list)
