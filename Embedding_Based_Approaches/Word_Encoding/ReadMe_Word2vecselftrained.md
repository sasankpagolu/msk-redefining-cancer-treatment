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