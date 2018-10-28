import networkx as nx
import numpy as np
import csv
import scipy

# Create an undirected graph
G = nx.read_edgelist('Cit-HepTh.txt', delimiter='\t', create_using=nx.Graph())

print("Nodes: ", G.number_of_nodes())
print("Edges: ", G.number_of_edges())

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

# Read test data
test_ids = list()
with open('test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

n_test = len(test_ids)

ids = np.hstack((train_ids, test_ids))
# Create a subgraph so that the computation is faster
subG=G.subgraph(ids)
print("Nodes: ", subG.number_of_nodes())
print("Edges: ", subG.number_of_edges())

def spectral_embedding(G,d):
    # Laplacian matrix
    n = len(G.nodes())
    A = nx.to_scipy_sparse_matrix(G)
    deg = scipy.sparse.csr_matrix.dot(A, np.ones(n))
    D = scipy.sparse.diags(deg)
    L = D - A
    
    # Spectral decomposition
    lam,V = scipy.sparse.linalg.eigsh(L, min(d,n - 1), sigma = -1)
    index = np.argsort(lam)
    lam,V = lam[index],V[:,index]

    # Spectral embedding 
    X = np.transpose(V)
    
    return lam,X   

#Kmeans
def dist(a, b):
    return np.linalg.norm(a-b)

def add(a, b):
    return a+b

def kmeans(k, points):
    choice = np.random.choice(len(points), k, False)
    clusters = [0 for p in points]
    centroids = [points[i] for i in choice]
    change = True
    epsilon = 0.1
    sse_list = []
    while change:
        change = False
        sse = 0
        for i, p in enumerate(points):
            d = dist(p, centroids[clusters[i]])
            sse += d**2
            for j, c in enumerate(centroids):
                if (dist(c, p) <= d):
                    clusters[i] = j
                    d = dist(c, p)
        sse_list.append(sse)
        for i in range(k):
            mean = np.zeros_like(points[0])
            n = 0
            for j, p in enumerate(points):
                if clusters[j] == i:
                    mean = add(mean, p)
                    n+= 1
            mean = 1/n * mean
            if ( dist(centroids[i] ,mean) > epsilon):
                change = True
                centroids[i] = mean
    return clusters, sse_list

# Spectral clustering
# G = graph
# d = dimension of the embedding (>=1)
# k = number of clusters (>=1)
def spectral_clustering(G,d,k):
    lam,X = spectral_embedding(G,d)
    points = np.transpose(X)
    km, sse = kmeans(k, points) 
    labels = set(km)
    Cest = [[i for i,j in enumerate(km) if j == l] for l in labels]
    return Cest

# perform the spectral clustering
d = 10
k = unique.size
Cest = spectral_clustering(subG,d,k)

# m is the matrix containing the proportions of each class in each cluster
m = np.zeros((28,28))
for i in range(unique.size):
    count = 0
    for k in Cest[i]:
        if k<n_train:
            count+=1
            p=unique.tolist().index(y_train[k])
            m[i][p]+=1
    m[i,:]=m[i,:]/count
      
# to each point in the test data, we give the probabilities of the cluster they are in
y_pred=np.zeros((n_test,unique.size))
for i in range(unique.size):
    for k in Cest[i]:
        if k>=n_train:
            p=k-n_train
            y_pred[p,:]=m[i,:]

# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = unique.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)