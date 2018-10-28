#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 19:19:16 2018

@author: jesusbm
"""

import numpy as np
import pandas as pd
import networkx as nx
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

########################################################TEXT

# Load data about each article in a dataframe
df = pd.read_csv("node_information.csv")
print(df.head())

# Extract the abstract of each training article from the dataframe
train_abstracts = list()
for i in train_ids:
    train_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])

# define the stemmer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
 
def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

# Create the training matrix. Each row corresponds to an article 
# and each column to a word present in at least 2 webpages and at
# most 50 articles. The value of each entry in a row is equal to 
# the frequency of that word in the corresponding article	
vec = TfidfVectorizer(tokenizer=stemming_tokenizer, stop_words='english', max_df=0.5, ngram_range=(1,2))
X_train_text = vec.fit_transform(train_abstracts)
X_train_text = X_train_text.toarray()

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

# Create the test matrix following the same approach as in the case of the training matrix
X_test_text = vec.transform(test_abstracts)
X_test_text = X_test_text.toarray()

########################################################GRAGPH

# Create a directed graph
G = nx.read_edgelist('Cit-HepTh.txt', delimiter='\t', create_using=nx.DiGraph())

print("Nodes: ", G.number_of_nodes())
print("Edges: ", G.number_of_edges())

# Create the training matrix. Each row corresponds to an article.
# Use the following 3 features for each article:
# (1) out-degree of node
# (2) in-degree of node
# (3) average degree of neighborhood of node
avg_neig_deg = nx.average_neighbor_degree(G, nodes=train_ids)
X_train_graph = np.zeros((n_train, 3))
for i in range(n_train):
	X_train_graph[i,0] = G.out_degree(train_ids[i])
	X_train_graph[i,1] = G.in_degree(train_ids[i])
	X_train_graph[i,2] = avg_neig_deg[train_ids[i]]

# Create the test matrix. Use the same 3 features as above
n_test = len(test_ids)
avg_neig_deg = nx.average_neighbor_degree(G, nodes=test_ids)
X_test_graph = np.zeros((n_test, 3))
for i in range(n_test):
	X_test_graph[i,0] = G.out_degree(test_ids[i])
	X_test_graph[i,1] = G.in_degree(test_ids[i])
	X_test_graph[i,2] = avg_neig_deg[test_ids[i]]

########################################################TRAINING

X_train = np.hstack((X_train_text, X_train_graph))
X_test = np.hstack((X_test_text, X_test_graph))

print("\nTrain matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

# Use logistic regression to classify the articles of the test set
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

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