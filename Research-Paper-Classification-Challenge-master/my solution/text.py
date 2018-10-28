#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 02:10:53 2018

@author: jesusbm
"""

import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import GridSearchCV 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# Load data about each article in a dataframe
df = pd.read_csv("node_information.csv")
print(df.info())

# show a graph with the data information
df_train=pd.read_csv("train.csv")
df_stats = df_train.groupby(['Journal']).count()
df_stats.plot(kind='bar', legend=False, grid=True, figsize=(15, 8))

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
    s = df.loc[df['id'] == int(i)]['title'].iloc[0]+' '
    if isinstance(df.loc[df['id'] == int(i)]['authors'].iloc[0],str):
        s=s+df.loc[df['id'] == int(i)]['authors'].iloc[0]+' '
    s=s+df.loc[df['id'] == int(i)]['abstract'].iloc[0]
    train_abstracts.append(s)

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
    s = df.loc[df['id'] == int(i)]['title'].iloc[0]+' '
    if isinstance(df.loc[df['id'] == int(i)]['authors'].iloc[0],str):
        s=s+df.loc[df['id'] == int(i)]['authors'].iloc[0]+' '
    s=s+df.loc[df['id'] == int(i)]['abstract'].iloc[0]
    test_abstracts.append(s)

# define the stemmer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
 
def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

"""    
# logistic regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=stemming_tokenizer, stop_words='english')),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=100))),
])
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}
"""

# svm
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=stemming_tokenizer, stop_words='english', min_df=7, max_df=1000, ngram_range=(1,2))),
    ('clf', SVC(kernel = 'linear', probability=True)),
])    
parameters = {
    'clf__estimator__C':[0.1, 1, 10],
}

clf = GridSearchCV(pipeline, parameters, cv=2, verbose=1)
clf.fit(train_abstracts, y_train)

print(clf.best_score_) 
print(clf.best_params_)

y_pred = clf.predict_proba(test_abstracts)

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