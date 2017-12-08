# Import Stattements
import gensim
import numpy as np
import html
import re
from nltk import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize.casual import TweetTokenizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
# Use these if using python 2.7
# import sys
# import codecs
# reload(sys)
# sys.setdefaultencoding('utf8')
import keras
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

import numpy as np
from tempfile import TemporaryFile
import pandas as pd
#import cPickle
import theano as th
from collections import defaultdict
import re
import sys
import os
#set this if using theano backend
#os.environ['KERAS_BACKEND']='theano'

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
			list_train.append(Text(array_splits[0],preprocess_string(array_splits[1])," "," "," "))

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
def preprocess_string(string):
	
	string = html.unescape(string)
	string = string.replace("\\n"," ")
	string = string.replace("_NEG","")
	string = string.replace("_NEGFIRST", "")
	string = re.sub(r"@[A-Za-z0-9_s(),!?\'\`]+", "", string) # removing any twitter handle mentions
	string = re.sub(r"#", "", string)
	string = re.sub(r"\*", "", string)
	string = re.sub(r"\'s", "", string)
	string = re.sub(r"\'m", " am", string)
	string = re.sub(r"\'ve", " have", string)
	string = re.sub(r"n\'t", " not", string)
	string = re.sub(r"\'re", " are", string)
	string = re.sub(r"\'d", " would", string)
	string = re.sub(r"\'ll", " will", string)
	string = re.sub(r",", "", string)
	string = re.sub(r"!", " !", string)
	string = re.sub(r"\(", "", string)
	string = re.sub(r"\)", "", string)
	string = re.sub(r"\?", " ?", string)
	string = re.sub(r'[^\x00-\x7F]',' ', string)
	# pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
	# string  = pattern.sub(' ',string)
	string = re.sub(r"\s{2,}", " ", string)
	return string #return remove_stopwords(string.strip().lower())
	
def remove_stopwords(string):
	split_string = \
	[word for word in string.split()
		if word not in stopwords.words('english')]

	return " ".join(split_string)


def LSTM_first_model(training_list):

	text_train = list();
	score_train = list();
	for text in training_list:
		text_train.append(text.text);
		score_train.append(int(text.category));

	score_train = to_categorical(score_train, num_classes=9)
	vocab_size = 500000
	max_len = 2000

	encoded_text_train	= [one_hot(d, vocab_size) for d in text_train]
	padded_train 	= pad_sequences(encoded_text_train,maxlen = max_len,padding = 'post')
	#Embedding Layes
	embedding_1 = Embedding(vocab_size,100, input_length=max_len)


	#LSTM Layers
	lstm_1 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, name='lstm1', return_sequences=True)
	lstm_2 = LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm2', return_sequences=True)
	lstm_3 = LSTM(16, dropout=0.2, recurrent_dropout=0.2, name='lstm3',return_sequences = True)
	#Dense Layers
	dense_1 = Dense(128, activation='relu', name='dense1')
	dense_2 = Dense(9, activation='sigmoid', name='dense2')

	def get_model():
		model = Sequential()
		model.add(embedding_1)
		# model.add(bi_lstm_1)
		# model.add(conv_1)
		# model.add(conv_2)
		# model.add(lstm_1)
		# model.add(lstm_2)
		model.add(lstm_3)
		model.add(Flatten())
		#model.add(dense_1)
		model.add(dense_2)
		#compile the model
		model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['categorical_crossentropy','acc'])
		# summarize the model
		print(model.summary())
		# fit the model
		return model

	#create the model
	
	X_train, X_test, Y_train, Y_test = train_test_split(padded_train, score_train, test_size = 0.2, random_state = 42, stratify=score_train)
	estimator = get_model()
	estimator.fit(X_train,Y_train,nb_epoch =5,batch_size=512,verbose=1)
	estimator.save('my_model_lstm_1.h5')  # creates a HDF5 file 'my_model.h5'
	probas = estimator.predict(X_test)
	pred_indices = np.argmax(probas, axis=1)
	classes = np.array(range(0,9))
	preds = classes[pred_indices]
	print('Log loss: {}'.format(log_loss(classes[np.argmax(Y_test, axis=1)], probas)))
	print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(Y_test, axis=1)], preds)))


# Function for running LSTM on top of Trained embeddings 
def embedding_lstm_glove(training_list,test_list):
	
	docs_train = list()
	score_train = list()
	total_dataset = list()
	for text in training_list:
		docs_train.append(text.text);
		score_train.append(int(text.category));
		total_dataset.append(text.text)
	
	score_train = to_categorical(score_train, num_classes=9)
	docs_test = list();
	# score_test = list();
	for text in test_list:
		docs_test.append(text.text);
	#	score_test.append(int(text.category));
		total_dataset.append(text.text)
#	print (total_daatset)

#	score_test = to_categorical(score_test, nb_classes=9)
	t = Tokenizer(lower=False)
	t.fit_on_texts(total_dataset)
	word_index = t.word_index
	print(t.document_count)
	vocab_size = len(t.word_counts)
	print(vocab_size)
	print (len(word_index))
	max_len = 300
	sequences_train = t.texts_to_sequences(docs_train)
	sequences_test = t.texts_to_sequences(docs_test)

	EMBEDDING_DIM = 100
	glove_train = gensim.models.Word2Vec.load('w2vmodeladdeddata').wv  # Be sure to have this dict in the same folder as the code
	# np.load('./Data/embeddingmap.npy',encoding = 'latin1');
	#glove_test = np.load('./Data/embeddingmap_test_wvec.npy',encoding = 'latin1')
#	print(len(glove_train))
	#print(len(glove_test.item()))

	embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))
	number_found =0
	number_not_found = 0
	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#	print(glove_train.item().keys())
	for word, i in word_index.items():
#	    
	    if word in glove_train.vocab:
	        
		embedding_vector = glove_train[word]
		embedding_matrix[i] = embedding_vector
	        number_found +=1
#		print('matched')
#		print(word)
	        continue
	    number_not_found+=1
#	    print('no match')
#	    print(word)

	print(number_found)
	print(number_not_found)
	
	padded_train 	= pad_sequences(sequences_train,maxlen = max_len,padding = 'post')
	padded_test		= pad_sequences(sequences_test,maxlen = max_len,padding = 'post')


	embedding_1 = Embedding(len(word_index) + 1,
	                            EMBEDDING_DIM,weights=[embedding_matrix],
	                            input_length=max_len,
	                            trainable=False)

	#LSTM Layers
	lstm_1 = LSTM(256, dropout=0.2, recurrent_dropout=0.2, name='lstm1', return_sequences=True)
	lstm_2 = LSTM(128, dropout=0.2, recurrent_dropout=0.2, name='lstm2', return_sequences=True)
	lstm_3 = LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm3')
	lstm_4 = LSTM(32, dropout=0.2, recurrent_dropout=0.2, name='lstm4',return_sequences = True)

	#Dense Layers
	dense_1 = Dense(200, activation='relu', name='dense1')
	dense_2 = Dense(9, activation='softmax', name='dense2')

	def get_model():
		model = Sequential()
		model.add(embedding_1)
#		model.add(lstm_1)
		model.add(lstm_2)
		model.add(lstm_3)
		model.add(dense_1)
		model.add(dense_2)
		#compile the model
		model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['categorical_crossentropy','acc'])
		# summarize the model
		print(model.summary())
		# fit the model
		return model

	#create the model
	estimator = get_model()
	data_train, data_test, labels_train, labels_test = train_test_split(padded_train,score_train,test_size=0.20, random_state=42)
	estimator.fit(data_train,labels_train,epochs =10,validation_data = (data_test,labels_test),batch_size=128,verbose=1)
	estimator.save('my_model_lstm_10_full.h5')  # creates a HDF5 file 'my_model.h5'
	probas = estimator.predict(data_test)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.array(range(0,9))
    preds = classes[pred_indices]
#        print('Log loss: {}'.format(log_loss(labels_test, probas)))
 #       print('Accuracy: {}'.format(accuracy_score(labels_test, preds)))
	estimator.fit(data_train,labels_train,validation_data = (data_test,labels_test),epochs=10,batch_size = 128,verbose=1)
	estimator.save('model_20_full.h5')
	probas = estimator.predict(data_test)
	pred_indices = np.argmax(probas, axis=1)
	classes = np.array(range(0,9))
	preds = classes[pred_indices]
#	print('Log loss: {}'.format(log_loss(labels_test, probas)))
#	print('Accuracy: {}'.format(accuracy_score(labels_test, preds)))
	
def embedding_lstm_wev_hierarchial(training_list,test_list):

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
	data = np.zeros((len(train_texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
	word_index = tokenizer.word_index
	for i, sentences in enumerate(train_documents):
	    for j, sent in enumerate(sentences):
	        if j< MAX_SENTS:
	            wordTokens = text_to_word_sequence(sent)
	            k=0
	            for _, word in enumerate(wordTokens):
	                if k<MAX_SENT_LENGTH:
	                    data[i,j,k] = tokenizer.word_index[word]
	                    k=k+1                    
	                    
	vocab_size = len(word_index)
	labels = to_categorical(np.asarray(train_labels))

	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	data = data[indices]
	labels = labels[indices]

	EMBEDDING_DIM = 100
    glove_train = gensim.models.Word2Vec.load('w2vmodeladdeddata').wv # np.load('./Data/embeddingmap.npy',encoding = 'latin1');
    #glove_test = np.load('./Data/embeddingmap_test_wvec.npy',encoding = 'latin1')
#       print(len(glove_train))
    #print(len(glove_test.item()))

    number_found =0
    number_not_found = 0
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#       print(glove_train.item().keys())
    for word, i in word_index.items():
#           embedding_vector = glove_train[word]
        if word in glove_train.vocab:
            # words not found in embedding index will be all-zeros.
            embedding_vector = glove_train[word]
            embedding_matrix[i] = embedding_vector
            number_found +=1
            continue
        number_not_found+=1
#           print('no match')
#           print(word)

    print(number_found)
    print(number_not_found)

	def get_model():
		input_sent = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
		embedding_layer = Embedding(len(word_index) + 1,
		                             EMBEDDING_DIM,weights=[embedding_matrix],
		                             input_length=MAX_SENT_LENGTH,
		                             trainable=False)
		emb_1 = embedding_layer(input_sent)
		l_lstm = Bidirectional(LSTM(100)) (emb_1)
		encoded_sentence = Model(input_sent, l_lstm)

		input_document = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
		encoded_document = TimeDistributed(encoded_sentence)(input_document)
		l_lstm_sent = Bidirectional(LSTM(100))(encoded_document)
		pen_ultimate = Dense(100,activation='sigmoid')(l_lstm_sent)
		preds = Dense(9, activation='softmax')(pen_ultimate)
		model = Model(input_document, preds)
		model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['categorical_crossentropy','acc'])
		return model        

	#create the model
	estimator = get_model()
	
	data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)
	estimator.fit(data_train,labels_train,validation_data = (data_test,labels_test), epochs =10,batch_size=64,verbose=1)
	estimator.save('model_10_hierarchial.h5')  # creates a HDF5 file 'my_model.h5'
	estimator.fit(data_train,labels_train,validation_data = (data_test,labels_test), epochs =10,batch_size=64,verbose=1)
	estimator.save('model_20_hierarchial.h5') 
#	estimator.fit(data,labels, epochs=10,batch_size = 64,verbose=1)
#	estimator.save('model_20_hierarchial.h5')
#	estimator.fit(data,labels,epochs=10,batch_size=64,verbose=1)
#	estimator.save('model_30_hierarchial.h5')
#	estimator.fit(data,labels,epochs=10,batch_size=64,verbose=1)
#	estimator.save('model_40_hierarchial.h5')
#	estimator.fit(data,labels,epochs=10,batch_size=32,verbose=1)
#	
	'''
	estimator.save('model_50_hierarchial.h5')
	probas = estimator.predict(data_test)
	pred_indices = np.argmax(probas, axis=1)
	classes = np.array(range(0,9))
	preds = classes[pred_indices]
	print('Log loss: {}'.format(log_loss(labels_test, probas)))
	print('Accuracy: {}'.format(accuracy_score(labels_test, preds)))
	estimator.save('my_model_lstm_10_full.h5')  # creates a HDF5 file 'my_model.h5'
        probas = estimator.predict(data_test)
        pred_indices = np.argmax(probas, axis=1)
        classes = np.array(range(0,9))
        preds = classes[pred_indices]
#        print('Log loss: {}'.format(log_loss(labels_test, probas)))
 #       print('Accuracy: {}'.format(accuracy_score(labels_test, preds)))
        estimator.fit(data_train,labels_train,validation_data = (data_test,labels_test),epochs=10,batch_size = 128,verbose=1)
        estimator.save('model_20_full.h5')
        probas = estimator.predict(data_test)
        pred_indices = np.argmax(probas, axis=1)
        classes = np.array(range(0,9))
        preds = classes[pred_indices]

	'''


training_variants_filename = "training_variants"
training_text_filename = "training_text"

test_filename = "stage2_test_text.csv"
test_variant = "stage2_test_variants.csv"

training_list = read_training_data(training_text_filename,training_variants_filename)
test_list = read_training_data(test_filename,test_variant)
print ("Done Reading File")
embedding_lstm_glove(training_list,test_list)
embedding_lstm_wev_hierarchial(training_list,test_list)



