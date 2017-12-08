''' README
Takes a text file as input and creates a numpy array of word2vec embeddings for each instance in the text file.
Also creates a dictionary mapping each word in the text file to a 400 dimensional vector embedding.

Steps to perform before running:
1) Make sure you have numpy package installed for python.
2) Download PubMed Word2vec 400 dimensional embeddings from this url (evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-ri.tar.gz)
3) UnZip the file and name vectors file as vectors.tsv and name labels file as vocab.tsv in the folder ri-3gram-400-tsv. 
4) The unzipped file along with the input text file must reside in the same directory as your code.
5) Input text file path to be set in inputtext variable in line#24
6) To get a written text file of word2vec vector embeddings for text file instances, set the output file 
   path in outputwrite variable in line#25
7) To store the vectors mapped to the words in our input file in a dictionary, set the output file 
   path in outputvocabdict variable in line#26
8) To store the vectors mapped to the text instances in our input file in a dictionary, set the output file 
   path in outputinstancedict variable in line#27
9) To create vectors for training text file, follow the steps 1-4 and run this file as it is.
10)To create vectors for test text file,comment the lines 24-27 and uncomment the lines 29-32 and
   run this file as it is.
'''

import numpy as np

inputtext='training_text'
outputwrite='Embeddings_w2vec_train.txt'
outputvocabdict='embeddingmap_train_wvec.npy'
outputinstancedict='output_train_wvec.npy'

#inputtext='stage2_test_text.csv'
#outputwrite='Embeddings_w2vec_test.txt'
#outputvocabdict='embeddingmap_test_wvec.npy'
#outputinstancedict='output_test_wvec.npy'

vectors='ri-3gram-400-tsv/vectors.tsv'
labels='ri-3gram-400-tsv/vocab.tsv'



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
dimension=400
'''
fp = open('/home/sasank/BigData/ri-3gram-400-tsv/output.txt')
for line in fp:
	values=line.split(' ')
	for word in values:
		dimension=dimension+1
fp.close()
'''
print dimension
print "\n"
#Creating dict
w2vec=dict()


print "mappping embeddings\n"
fp1=open(vectors)
fp2= open(labels)
i=0

for line in fp1:
	vocabs=fp2.readline().split('\t')
	values=line.rstrip('\n')
	w2vec[vocabs[0]]=values
	i=i+1
	if i%100000 == 0:
		print i
	
fp1.close()
fp2.close()

print "mapped embeddings\n"


embeddingmap=dict()

fp = open(inputtext)
next(fp)


i=0
for line in fp:
	values=line.split('||')
	docs=values[1].split(' ')
	for word in docs:
		word=word.rstrip('.|,|;|:|\'|\"')
		word=word.lstrip('\'|\"')
		x=w2vec.get(word)
		
		if x is not None:
			y = np.fromstring(x, dtype=np.float, sep="\t")
			embeddingmap[word]=y
			
		
	i=i+1
	if i%50==0:
		print i



fp.close()




print "Embedding_map done"


fp = open(inputtext)
wfp = open(outputwrite, 'w')
out=[]
next(fp)
i=0
for line in fp:
	allsum=np.zeros((dimension,),dtype="float32")
	count=0
	values=line.split('||')
	docs=values[1].split(' ')
	for word in docs:
		word=word.rstrip('.|,|;|:|\'|\"')
		word=word.lstrip('\'|\"')
		x=embeddingmap.get(word)
		
		if x is not None:			
			allsum=np.add(allsum, x)
			count=count+1
		
	if count!=0:	
		wfp.write("%s\n" % np.divide(allsum,count))
		out.append(np.divide(allsum,count))
	else:
		wfp.write("%s\n" % allsum)
		out.append(allsum)
#	i=i+1
#	print i
np.save(outputvocabdict,embeddingmap)
np.save(outputinstancedict,out)
wfp.close()
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
