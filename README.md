------- README --------

The following repository contains the codes developed for the Kaggle Task: "Personalized Medicine: Redefining Cancer Treatment".
In this project our goal was to use Machine Learning models to accurately classify genetic mutations from expert 
annotated knowledge bases and text based clinical literature into a set of predefined classes.
For achieving this goal we tried a variety of approaches from using N-grams and TF-IDF,
word embedding based approaches to deep learning based approaches. We obtained the best results with training Word2Vec
on our dataset and using LightGBM classifier. The approach gave a log-loss of 1.98 on the Kaggle private leaderboard and
was ranked number 1 from 1386 total teams. Based on the results that we obtained by trying the various
methods we conclude that if the size of the dataset is good enough,training word embeddings can significantly improve the performance.

------- 
