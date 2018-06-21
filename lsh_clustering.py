import numpy as np
#import MinHash
from datasketch import MinHash, MinHashLSH
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import pandas as pd 

data = pd.read_csv("training160000.csv", delimiter=',', encoding="utf-8")

dataset = data.head(50)
#text = dataset.iloc[:, 5].values

documents = list(dataset.iloc[:, 5].values)
n_samples = len(documents)

#MinHash all string
n_perms = 128


#Min hash all string
minhash_values = np.zeros((n_samples, n_perms), dtype='uint64')
minhashes = []

for index, document in enumerate(documents):
    
    minhash1 = MinHash(num_perm=n_perms)
    shingles = [document[i:i+9] for i in range(len(document))][:-9]

    for grams in shingles:
        minhash1.update(grams.encode('utf8'))
    
    minhash_values[index, :] = minhash1.hashvalues
    minhashes.append(minhash1.hashvalues)

#check all min hash if they are thesame
#print(np.array_equal(minhashes[0],minhashes[1]))
#print("He")
#print(minhash_values[1])

#compute cluster
clusterer = KMeans(n_clusters=8)
#clusters = clusterer.fit_predict(minhash_values)
#clusters = clusterer.fit(minhash_values)
clusters = clusterer.fit(minhashes)#
#print("Clusters is", clusters)

centroids = clusterer.cluster_centers_
#print("Centroids is", centroids)

labels = clusterer.labels_

print("labels is " , labels)

#print("lenmgth of min_hash_values", minhashes[2])
colors = ["green","red", "cyan.","blue"]


pca = PCA(n_components=2).fit(minhashes)
pca_2d = pca.transform(minhashes) 
#print(minhashes)
#print(pca_2d)

#for i in range(len(minhashes)):
#    print("cordinates : ", pca_2d[i], " - Label : ", labels[i])


for i in range(len(minhash_values)):
    #print("cordinate : " ,minhash_values[i], " - Label : ", labels[i])
    print("twwet  : " ,i, "  ",   documents[i] , " "  " -  ", labels[i])
    #plt.plot(pca_2d[:, 0], pca_2d[:, 1], colors[labels[i]], markersize =10)
    #plt.plot(pca_2d[:, 0], pca_2d[:, 1], colors=[labels[i]], markersize =10)

#plt.scatter(pca_2d[:, 0], pca_2d[:, 1])
#plt.scatter(centroids[:,0], centroids[:,1], color='blue', marker="X", s=150, linewidths=5, zorder = 10)
#plt.show()

'''
for d in set1:
    minhash1.update(d.encode('utf8'))


for index, string in enumerate(data):
    minhash = MinHash(num_per=n_perms)

    for gram in ngrams(string, 3):
        minhash.update(" ".join(gram).encode('utf-8'))
    
    minhash_values[index, :] = minhash.hashvalues

#compute cluster 
clusterer = KMeans(n_clusters=3)
clusters = clusterer.fit_predict(minhash_values)

print(clusters)
'''