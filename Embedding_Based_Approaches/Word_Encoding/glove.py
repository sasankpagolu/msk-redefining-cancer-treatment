''' README
Takes a text file as input and creates a numpy array of glove embeddings for each instance in the text file.
Also creates a dictionary mapping each word in the text file to a 300 dimensional vector embedding.

Steps to perform before running:
1) Make sure you have numpy package installed for python.
2) Download glove 300 dimensional embeddings from this url (nlp.stanford.edu/data/glove.840B.300d.zip)
3) UnZip the file and name it as 'glove.840B.300d.txt'
4) The unzipped file along with the input text file must reside in the same directory as your code.
5) Input text file path to be set in inputtext variable in line#24
6) To get a written text file of glove vector embeddings for text file instances, set the output file 
   path in outputwrite variable in line#25
7) To store the vectors mapped to the words in our input file in a dictionary, set the output file 
   path in outputvocabdict variable in line#26
8) To store the vectors mapped to the text instances in our input file in a dictionary, set the output file 
   path in outputinstancedict variable in line#27
9) To create glove vectors for training text file, follow the steps 1-4 and run this file as it is.
10)To create glove vectors for test text file,comment the lines 24-27 and uncomment the lines 28-31 and
   run this file as it is.
'''

import numpy as np

inputtext='training_text'
outputwrite='Embeddings_glove_train.txt'
outputvocabdict='embeddingmapglove_train.npy'
outputinstancedict='output_train_glove.npy'
#inputtext='stage2_test_text.csv'
#outputwrite='Embeddings_glove_test.txt'
#outputvocabdict='embeddingmapglove_test.npy'
#outputinstancedict='output_test_glove.npy'

glovedictinput='glove.840B.300d.txt'


embeddings=[]

print "Find no_of_instances"
no_of_instances=0
fp = open(inputtext)
next(fp)
for line in fp:
	no_of_instances=no_of_instances+1
fp.close()
print no_of_instances
print "\n"
print "Finding the dimension of the embeddings"
dimension=300

print dimension
print "\n"
#Creating dict
glove=dict()


print "mappping embeddings\n"
fp=open(glovedictinput)
i=0

for line in fp:
	
	values=line.rstrip('\n').split(' ')
	word=values[0]
	word=word.rstrip('.|,|;|:|\'|\"|)|}|]')
	word=word.lstrip('\'|\"|(|[|{')
	glove[word]=values[1:len(values)]
	i=i+1
	if i%100000==0:
		print i

fp.close()

print "mapped embeddings\n"

embeddingmap=dict()

fp = open(inputtext)
next(fp)


i=0
for line in fp:
	values=line.split('||')
	docs=values[1].split(' ')
	for word in docs:
		temp=word
		word=word.rstrip('.|,|;|:|\'|\"|)|}|]')
		word=word.lstrip('\'|\"|(|[|{')
		x=glove.get(word)
		if x is not None:
			embeddingmap[word]=x
			#print word
		
	i=i+1
	if i%50==0:
		print i


fp.close()



print "Embedding_map done"

output_glove_train=[]
fp = open(inputtext)
wfp = open(outputwrite, 'w')
next(fp)
i=0
for line in fp:
	allsum=np.zeros((dimension,),dtype="float32")
	count=0
	values=line.split('||')
	docs=values[1].split(' ')
	for word in docs:
		word=word.rstrip('.|,|;|:|\'|\"|)|}|]')
		word=word.lstrip('\'|\"|(|[|{')
		x=embeddingmap.get(word)
		
		if x is not None:
			y=[float(v) for v in x]				
			allsum=np.add(allsum, y)
			count=count+1
		
	if count!=0:	
		wfp.write("%s\n" % np.divide(allsum,count))
		output_glove_train.append(np.divide(allsum,count))
	else:
		wfp.write("%s\n" % allsum)
		output_glove_train.append(allsum)
	i=i+1
	if i%50==0:
		print i
	
np.save(outputvocabdict,embeddingmap)
np.save(outputinstancedict,output_glove_train)
print "done"

'''
print "Finding all words in corpus"
fp = open('/home/sasank/BigData/training_text')
next(fp)
words=[]
for line in fp:
	values=line.split('||')
	docs=values[1].split(' ')
	for word in docs:
		word=word.rstrip('.|,|;|\s|:|\'|\"')
		word=word.lstrip('\'|\"')
		if(len(word)>0):
			words.append(word)
fp.close()
print "Found all words in corpus\n"
'''

'''

print "finding only the necessary vocab from word2vec corpus"

fp = open('/home/sasank/BigData/ri-3gram-400-tsv/vocab.tsv')
indices=[]
vocabs=[]
i=0
for line in fp:
	values=line.split(' ')
	if values[0] in words:
		indices.append(i)
		vocabs.append(values[0])
	i=i+1
	print i
fp.close()
print "found only the necessary vocab from word2vec corpus\n"


print "Embeddings Finding\n"
embeddings=zeros(no_of_instances,dimension)
'''
'''
for x in d1.keys():
	print x
	print len(d1.get(x))
	print (d1.get(x))
	exit()
'''
