README

The following repository contains the codes developed for the Kaggle Task: "Personalized Medicine: Redefining Cancer Treatment".
In this project our goal was to use Machine Learning models to accurately classify genetic mutations from expert 
annotated knowledge bases and text based clinical literature into a set of predefined classes.
For achieving this goal we tried a variety of approaches from using N-grams and TF-IDF,
word embedding based approaches to deep learning based approaches. We obtained the best results with training Word2Vec
on our dataset and using LightGBM classifier. The approach gave a log-loss of 1.98 on the Kaggle private leaderboard and
was ranked number 1 from 1386 total teams. Based on the results that we obtained by trying the various
methods we conclude that if the size of the dataset is good enough,training word embeddings can significantly improve the performance.

Kaggle Competition Link: https://www.kaggle.com/c/msk-redefining-cancer-treatment

Project Description: In this project, we aim to classify mutations that cause cancer by examining medical text records.  A cancer tumor can have thousands of genetic mutations. The challenge is to distinguish the mutations that contribute to tumor growth from the neutral mutations. This will help in diagnosing cancer causing mutations early and treat the patient’s tumor at the preliminary stage. Not only this, an automated system of this kind can help eliminate the effort and time invested by clinical pathologist in manually reviewing and classifying every single genetic mutation based on evidence from text-based clinical literature. We have used machine learning and deep learning models in tandem with natural language processing techniques such as word embeddings, tfidf for classifying texts. We deeply investigated the utility of deep learning models like LSTMs and CNNs for this task. After extensive feature engineering, we have used several machine learning classifiers like Gradient Boosting, Support Vector Machine, Logistic Regression and Artificial Neural networks to classify the extracted features.

Repository Description/ How to run the Codes: 
In the following repository we have folders for each of the approaches:
  1. Embedding Based
  2. TF-IDF 
  3. Deep Learning
  In each of these folders, we provide the codes for the apporaches along with Readme files to run these codes.
 The repository also contains Visualization folder which consists of a R script for visualizing the data and the Submission folder contains all the Kaggle Submission Files.  

Python Packages Required:
numpy, keras, tensorflow, Xgboost, scikit learn, pandas, LightGBM, gensim, h5py, pickle, os, nltk, html, theano, codecs

Data Link: https://www.kaggle.com/c/msk-redefining-cancer-treatment/data
Data Sample: 
1. Training_Variant File
-----------------------------------
ID | Gene | Variation | Class

1  | FAM58A | Truncating Mutation | 1   
-----------------------------------
2. Training_Text File
-----------------------------------
ID | Text

1  | Cyclin-dependent kinases (CDKs) regulate a variety of fundamental cellular processes ........
-----------------------------------

Drive Folder: https://drive.google.com/drive/folders/1Fdg67U6eylgcnK5K8yZydyRLTfvhQmnL?usp=sharing
The drive folder contains the following:
1. Trained Word2Vec Model
2. Vector.tsv and Labels.csv required for visualization (projector.tensorflow.org)




