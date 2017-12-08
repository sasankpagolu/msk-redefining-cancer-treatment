#keras imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential

#scikit learn imports
from keras.preprocessing.text import one_hot
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import log_loss, accuracy_score
from sklearn.utils import class_weight

#general python imports
import gensim
import numpy as np
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
	with open (training_text, encoding='utf8') as file_input:
		for line in file_input:
			line = line.strip()
			array_splits = line.split('||')
			if len(array_splits)<2 :
				continue
			list_train.append(Text(array_splits[0],(array_splits[1])," "," "," "))

	i = 0
	k = 0
	with open(training_variants) as file_input:
		for line in file_input:
			line = line.strip()
			k = k+1
			if (k==1):
				continue
			array_splits = line.split(',')
			list_train[i].gene = array_splits[1]
			list_train[i].mutation = array_splits[2]
			list_train[i].category = int(array_splits[3])-1
			i = i+1

	return list_train

def CNN_LEARN(training_list):
	text_train = [];
	score_train = [];
	for text in training_list:
		text_train.append(text.text);
		score_train.append(int(text.category));
	
	vocab_size = 5000
	max_len = 500
	
	texts = text_train
	myclass_weight = class_weight.compute_class_weight('balanced', np.unique(score_train), score_train)
	score_train = to_categorical(score_train, num_classes=9)
	MAX_NO_WORDS = 1000
	
	#this will restrict the index only top 1000 words in document
	tokenizer = Tokenizer(lower=False)
	
	#fit the tokenizer on texts
	tokenizer.fit_on_texts(texts)
	word_index = tokenizer.word_index
	
	#convert the sentecess in text to sequences
	sequences = tokenizer.texts_to_sequences(texts)
	encoded_text_train = [one_hot(d, vocab_size) for d in texts]
	
	#check the number of unique tokens
	#word_index = tokenizer.word_index
	#print('Found %s unique tokens.' % len(word_index))

	#keep the sequence lengths of fixed size, if > then truncate, else pad with 0
	#MAX_SEQUENCE_LENGTH = 300
	data = pad_sequences(encoded_text_train, maxlen=max_len)
	#assume x_train, x_test, y_train, y_test variables contain the required data
	#populate the data accordingly.

	#load the word embeddings from the pretrained word vectors and get a dictionary
	embeddings_index = {}
	
	embedding_vector_length = 100
	'''
	embedding_matrix = np.zeros(len(word_index) + 1, EMBEDDING_DIM)
	for word, i in word_index.items():
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
			# words not found in the embedding index will be zeros
			embedding_matrix[i] = embedding_vector

	'''
	
	#split the data into training and test set	
	x_train, x_test, y_train, y_test = train_test_split(data, score_train, test_size = 0.2, random_state = 42, stratify=score_train)
	
	#load pretrained word model
	w2v = gensim.models.Word2Vec.load('../embeddings/w2vmodeladdeddata').wv
	#word_index = w2v.word_index
	number_found = 0
	number_not_found = 0

	#create the emeddinig matrix based on the words in vocab in and the embdedding vecrtors
	embedding_matrix = np.zeros((len(word_index) + 1, embedding_vector_length))
	for word, i in word_index.items():
		if word in w2v.vocab:
			embedding_vector = w2v[word]
			embedding_matrix[i] = embedding_vector
			number_found += 1
			continue
		number_not_found += 1

	#create an embedding layer with the embeddings created above
	embedding_layer = Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length = max_len, trainable=False)
	##print(x_train.shape)
	print('Number of embeddings found: {}'.format(number_found))
	model = Sequential()
	
	
	#This below model is a cnn architecture implemented according to the Medical Text Classification using CNN paper
	#model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_len))
	'''
	model.add(Conv1D(256, 5, activation='relu', padding='same'))
	model.add(Conv1D(256, 5, activation='relu', padding='same'))
	model.add(MaxPooling1D(5))
	model.add(Conv1D(256, 5, activation='relu', padding='same'))
	model.add(Conv1D(256, 5, activation='relu', padding='same'))
	model.add(MaxPooling1D(5))
	model.add(Dropout(0.5))
	#model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm2', return_sequences=False))

	#input=model.layers[7].output.get_shape(),
	model.add(Flatten())
	model.add(Dense(128, activation='relu', name='dense1'))
	model.add(Dropout(0.5))
	model.add(Dense(9, activation='softmax', name='dense2'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	'''

	#model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_len))

	#This is the CNN architecture that gave the best results. It has two sets of two 1D Conv layers followed by a MaxPooling Layer
	#The number of fileters is restricted to 100 and two dropout layers are used to prevent overfitting
	model.add(embedding_layer)
	model.add(Conv1D(filters=100, kernel_size=4, activation='relu'))
	model.add(Conv1D(filters=100, kernel_size=4, activation='relu'))
	
	model.add(MaxPooling1D(3))
	#model.add(Flatten())
	model.add(Conv1D(filters=100, kernel_size=4, activation='relu'))
	#model.add(MaxPooling1D(3))
	#model.add(Flatten())
	model.add(Conv1D(filters=100, kernel_size=4, activation='relu'))
	model.add(MaxPooling1D(3))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', name='dense1'))
	model.add(Dropout(0.5))
	model.add(Dense(9, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy'])
	
	print(model.summary())
#	print(y_train.shape)

	#Trained the built model on the training data
	model.fit(x_train, y_train, nb_epoch=5, batch_size=64, class_weight=myclass_weight)

	#Evaluate the scores of the model
	scores = model.evaluate(x_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))
	probas = model.predict(x_test)
	pred_indices = np.argmax(probas, axis=1)
	classes = np.array(range(0,9))
	preds = classes[pred_indices]
	#model.save('../models/cnn_model4.h5')
	print('Log loss: {}'.format(log_loss(classes[np.argmax(y_test, axis=1)], probas)))
	print('Accuracy: {}'.format(accuracy_score(classes[np.argmax(y_test, axis=1)], preds)))	
	
training_variants_filename = "training_variants"
training_text_filename = "training_text"
training_list = read_training_data(training_text_filename,training_variants_filename)
print ("Done Reading File")
print (training_list[0].category)
CNN_LEARN(training_list)