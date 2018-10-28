#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 16:03:32 2018

@author: jesusbm
"""

import numpy as np
import pandas as pd
from gensim.models import doc2vec
from collections import namedtuple
import csv
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.multiclass import OneVsRestClassifier

# Load data about each article in a dataframe
df = pd.read_csv("node_information.csv")
print(df.head())

# Read training data
train_ids = list()
y_train = list()
with open('train.csv', 'r') as f:
    next(f)
    for line in f:
        t = line.split(',')
        train_ids.append(t[0])
        y_train.append(t[1][:-1])

n_train = len(train_ids)
unique = np.unique(y_train)
print("\nNumber of classes: ", unique.size)

# Extract the abstract of each training article from the dataframe
train_abstracts = list()
for i in train_ids:
    train_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])

# convert the data into a proper input for the Doc2Vec model
docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i in range(n_train):
    words = train_abstracts[i].lower().split()
    tags = [train_ids[i]]
    #tags=[y_train[i]]
    docs.append(analyzedDocument(words, tags))
    
# Read test data
test_ids = list()
with open('test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

# Extract the abstract of each test article from the dataframe
n_test = len(test_ids)
test_abstracts = list()
for i in test_ids:
    test_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])
    
# convert the data into a proper input for the Doc2Vec model   
for i in range(n_test):
    words = test_abstracts[i].lower().split()
    tags = [test_ids[i]]
    docs.append(analyzedDocument(words, tags))
 
# Doc2Vec model
model = doc2vec.Doc2Vec(min_count=1, window=10, vector_size=100, sample=1e-4, negative=5, workers=2, epochs=20)
model.build_vocab(docs)
model.train(docs, epochs=model.iter, total_examples=model.corpus_count)
# Build doc2vec vectors
x_train = []
x_test= []
for i in range(n_train):
    x_train.append(model.docvecs[i])
for i in range(n_test):
    x_test.append(model.docvecs[n_train+i])

# classification
pipeline = Pipeline([
        ('clf', OneVsRestClassifier(SVC(probability = True), n_jobs=1)),
    ])
parameters = {'kernel':('linear', 'rbf'), 
              'C':[0.1, 1, 10]}

clf = GridSearchCV(pipeline, parameters, cv=2, verbose=1)
clf.fit(x_train, y_train)

print(clf.best_score_) 
print(clf.best_params_)

y_pred = clf.predict_proba(x_test)

# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)  
    


                

                
                
