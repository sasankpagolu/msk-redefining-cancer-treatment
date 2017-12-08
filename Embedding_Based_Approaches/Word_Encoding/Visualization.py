'''
README: Creates Vectors and labels tsv files from your trained model for visualization purpose.
install all the dependencies
Put your trained model path in modelpath varaible in line# 17
creates vectors.tsv and labels.tsv files which can be uploaded and visualized in (projector.tensorflow.org)
'''

import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from gensim.models.keyedvectors import KeyedVectors
import sys
reload(sys)
sys.setdefaultencoding('utf8')

modelpath='w2vmodeladdeddata'
# define training data
model = gensim.models.Word2Vec.load(modelpath)
# fit a 2d PCA model to the vectors
wfp=open('vectors.tsv','w')
wfp1=open('labels.tsv','w')
for w in model.wv.vocab:
	wfp.write("%s\n" % '\t'.join(str(e) for e in model[w]))
	#print w
	wfp1.write("%s\n" % w)
