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