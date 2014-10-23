## Dara Elass
## Unsupervised learning challenges (KMeans)

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import pymongo
from pymongo import MongoClient
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.preprocessing import scale
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from operator import itemgetter
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import os
import HTMLParser
import operator
import pandas as pd

# set up; get data
client = MongoClient()
twitterdata = client.projectfletcher.twitterdata
h = HTMLParser.HTMLParser()
tweets = [h.unescape(tweet['text']) for tweet in twitterdata.find({},{"text":1,"_id":0})][0:5000]

############################# Challenge 1 #############################

# Cluster sentences with K-means. If you have your own Fletcher test data, get sentences out and cluster them.
# If not, cluster the tweets you gathered during the Twitter API challenge.
# For each cluster, print out the sentences, try to see how close the sentences are.
# Try different K values and try to find a K value that makes the most sense (the sentences look like they do form a meaningful cluster).

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(tweets)

k = 5 # number of clusters
print 'the number of clusters in this analysis:', k
print
model = KMeans(k).fit(X)
clusters = model.predict(X) # array of 5000 that says which cluster each tweet belongs to
dict_tweets = defaultdict(list)
for i, cluster in enumerate(clusters):
    dict_tweets[cluster].append(tweets[i])

for cluster_id, tweets in dict_tweets.iteritems():
    print '------CLUSTER %s ---------' % cluster_id
    for tweet in tweets[:10]:
        print tweet
        print
    print '--------------------------'
    print

############################# Challenge 2 #############################

# Draw the inertia curve over different k values. (Sklearn KMeans class has an inertia_ attribute)

inertia = []
m = 15
for k in range(1,m+1):
    inertia.append(KMeans(k).fit(X).inertia_)
k = [i+1 for i in range(m)]
plt.plot(k, inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.title('Inertia of K')
plt.savefig('inertias.png')
os.system('open inertias.png')

############################# Challenge 3 #############################

# For each cluster, find the sentence closest to the centroid of the
# (You can learn sklearn.metrics.pairwise_distances5 or scipy.spatial.distance5
# (check pdist, cdist, and euclidean distance) to find distances to the centroid.
# KMeans has a cluster_centroids_ attribute. This sentence (closest to centroid) is now the name of the cluster.
# For each cluster, print the representative sentence, and print 'N people expressed a similar statement',
# or something like that relevant to your dataset.
# (This is very close to what amazon used to do in the reviews section up to a year ago.)

cluster_centers = KMeans(k).fit(X).cluster_centers_ 
min_dist = {}
cluster_sizes = {}
cluster_names = []
for i in range(k):
    X_this_cluster = X[clusters == i]
    euc_dist = np.array(pairwise_distances(X_this_cluster, cluster_centers[i], metric='euclidean')) # 5000
    id_of_closest = euc_dist.argmin()
    num_ppl_in_cluster = len(dict_tweets[i])
    cluster_sizes[i] = num_ppl_in_cluster
    print num_ppl_in_cluster,'People expressed a similar statement to:'
    print dict_tweets[i][id_of_closest]
    cluster_names.append(dict_tweets[i][id_of_closest])
    print

# Find the biggest 3 clusters, and print their representative sentences
# (This is close to what amazon is doing now in the reviews section, except they choose the sentence
# from the most helpful review instead of closest to center)

sorted_cluster_sizes = sorted(cluster_sizes.items(), key = operator.itemgetter(1), reverse=True)
words = ['first','second','third']
for j in range(3):
    biggest_cluster_number = sorted_cluster_sizes[j][0]
    print 'the representative sentence for the ' + words[j] + ' biggest cluster (cluster '+str(biggest_cluster_number) + ' is:'
    print cluster_names[biggest_cluster_number]
    print

############################# Challenge 4 #############################

# Calculate the tf-idf of each word in each cluster (think of all sentences of a cluster together as a document).
# Represent each cluster with the top 1, or top 2 or... to 5 tf-idf words. For each cluster,
# print the name (keywords) of the cluster, and "N statements" in the cluster (N is the size of the cluster)

stop_words_cluster = stopwords.words('english')
stop_words_cluster.append("http")

#ngrams = 1

vectorizer_clusters = TfidfVectorizer(stop_words=stop_words_cluster)

for i in range(k):
    cluster_tfidf = {}
    X_cluster = vectorizer_clusters.fit_transform(dict_tweets[i])
    feature_names = vectorizer_clusters.get_feature_names()
    print 'cluster: ',i, 'size: ',X_cluster.shape[0],'length of feature names:',len(feature_names)
    for x in X_cluster.nonzero()[1]:
        cluster_tfidf[feature_names[x]] = X_cluster[(0,x)]
    cluster_tfidf_sorted = sorted(cluster_tfidf.items(), key = operator.itemgetter(1), reverse=True)
    for word, tfidf in cluster_tfidf_sorted[:5]:
        print '%20s %g' % (word,tfidf)
    print

#ngrams = 2
vectorizer_clusters = TfidfVectorizer(ngram_range = (2,2),stop_words=stop_words_cluster)

for i in range(k):
    cluster_tfidf = {}
    X_cluster = vectorizer_clusters.fit_transform(dict_tweets[i])
    feature_names = vectorizer_clusters.get_feature_names()
    print 'cluster: ',i, 'size: ',X_cluster.shape[0],'length of feature names:',len(feature_names)
    for x in X_cluster.nonzero()[1]:
        cluster_tfidf[feature_names[x]] = X_cluster[(0,x)]
    cluster_tfidf_sorted = sorted(cluster_tfidf.items(), key = operator.itemgetter(1), reverse=True)
    for word, tfidf in cluster_tfidf_sorted[:5]:
        print '%20s %g' % (word,tfidf)
    print
        
############################# Challenge 5 #############################

# Same as the previous challenge, but this time, calculate tf-idf only for nouns (NN tag)
# and build keyword(s) with nouns. (This is close to what amazon switched to last year,
# before settling into the current design). (They would show five nouns,
# you would click on one and it would show sentences - linked to the reviews- that were related to that noun.)

stop_words_cluster = stopwords.words('english')
stop_words_cluster.append("http")
vectorizer_clusters_tags = TfidfVectorizer(stop_words=stop_words_cluster)
cluster_tfidf_tags = {}

for i in range(k):
    cluster_tags = []
    print 'cluster:',i
    cluster_text = TextBlob(','.join(dict_tweets[i]))
    for tags in cluster_text.tags:
        if tags[1] == 'NN':
            cluster_tags.append(tags[0])
    X_cluster_tags = vectorizer_clusters_tags.fit_transform(cluster_tags)
    print X_cluster_tags
    feature_names = vectorizer_clusters_tags.get_feature_names()
    print 'cluster: ',i, 'size: ',X_cluster_tags.shape[0],'length of feature names:',len(feature_names)
    for x in X_cluster_tags.nonzero()[1]:
        cluster_tfidf_tags[feature_names[x]] = X_cluster_tags[(0,x)]
    cluster_tfidf_sorted = sorted(cluster_tfidf_tags.items(), key = operator.itemgetter(1), reverse=True)
    for word, tfidf in cluster_tfidf_sorted[:5]:
        print '%20s %g' % (word,tfidf)
    print

############################# Challenge 6 #############################

# Cluster the same data with MiniBatchKMeans.
# MiniBatchKMeans is a fast way to apply K-means to large data without much loss -- The results are very similar. Instead of using EVERY single point to find the new place of the centroid,
# MiniBatch just randomly samples a small number (like 100) in the cluster to calculate the new center.
# Since this is usually very close to the actual center, the algorithm gets there much faster.
# Try it and compare the results.

stop_words_cluster = stopwords.words('english')
stop_words_cluster.append("http")
vectorizer_mini = TfidfVectorizer(stop_words=stop_words_cluster)
X = vectorizer_mini.fit_transform(tweets)

k = 5 # number of clusters
print 'the number of clusters in this analysis:', k
print
model = MiniBatchKMeans(k).fit(X)
clusters = model.predict(X) # array of 5000 that says which cluster each tweet belongs to
dict_tweets = defaultdict(list)
for i, cluster in enumerate(clusters):
    dict_tweets[cluster].append(tweets[i])

# ngrams = 1
for i in range(k):
    mini_tfidf = {}
    X_mini = vectorizer_mini.fit_transform(dict_tweets[i])
    feature_names = vectorizer_mini.get_feature_names()
    print 'cluster: ',i, 'size: ',X_mini.shape[0],'length of feature names:',len(feature_names)
    for x in X_mini.nonzero()[1]:
        mini_tfidf[feature_names[x]] = X_mini[(0,x)]
    cluster_tfidf_sorted = sorted(mini_tfidf.items(), key = operator.itemgetter(1), reverse=True)
    for word, tfidf in cluster_tfidf_sorted[:5]:
        print '%20s %g' % (word,tfidf)
    print

# ngrams = 2
vectorizer_mini = TfidfVectorizer(ngram_range=(2,2),stop_words=stop_words_cluster)

for i in range(k):
    mini_tfidf = {}
    X_mini = vectorizer_mini.fit_transform(dict_tweets[i])
    feature_names = vectorizer_mini.get_feature_names()
    print 'cluster: ',i, 'size: ',X_mini.shape[0],'length of feature names:',len(feature_names)
    for x in X_mini.nonzero()[1]:
        mini_tfidf[feature_names[x]] = X_mini[(0,x)]
    cluster_tfidf_sorted = sorted(mini_tfidf.items(), key = operator.itemgetter(1), reverse=True)
    for word, tfidf in cluster_tfidf_sorted[:5]:
        print '%20s %g' % (word,tfidf)
    print

############################# Challenge 7 #############################

# Switch the init parameter to "random" (instead of the default kmeans++) and plot the inertia curve
# for each of the n_init values for K-Means: 1, 2, 3, 10 (n_init is the number of different runs to try
# with different random initializations)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(tweets)
ns = [1,2,3,10]
inertia = []
for i in ns:
    inertia.append(KMeans(i, init='random').fit(X).inertia_)
plt.plot(ns, inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.title('Inertia of K - Random')
plt.savefig('inertias_random.png')
os.system('open inertias_random.png')

############################# Challenge 8 #############################

# Download this dataset on the purchase stats from clients of a wholesale distributor.
# Cluster the clients based on their annual spending features
# (fresh, milk, grocery, frozen, detergents_paper, delicatessen). Remember to scale the features
# before clustering. After finding a reasonable amount of clusters, for EACH cluster,
# plot the histogram for every single feature: FRESH, MILK, GROCERY, FROZEN, DETERGENTS_PAPER,
# DELICATESSEN, CHANNEL, REGION. Is there a natural way to characterize each cluster?
# How would you describe each cluster to the wholesale distributor if you were working for them?

labels = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']
grocery_data = pd.read_csv('Wholesale customers data.csv')[labels] # 6 features
grocery_data_array = np.array(grocery_data)
grocery_data_scaled = scale(grocery_data_array.astype(np.float64))
print 'number of customers:', len(grocery_data_scaled)

k = 8
model = KMeans(k).fit(grocery_data_scaled)
clusters = model.predict(grocery_data_scaled) 
dict_groceries = defaultdict(list)
for i, cluster in enumerate(clusters):
    dict_groceries[cluster].append(grocery_data_scaled[i])

# remove previous histograms
filelist = [ f for f in os.listdir(".") if f.endswith("groceries.png") ]
print 'removing %i files' %len(filelist)
for f in filelist:
    os.remove(f)

plt.figure(figsize=(60,60))
for i in range(k):
    print 'number of customers in cluster %i is :' %i, len(dict_groceries[i])
    dict_groceries[i] = np.transpose(dict_groceries[i]) # each line/array is a feature
    print dict_groceries[i].shape
    for j in range(6): # number of features
        #print list(dict_groceries[i][j])
        #print i*6+j+1
        plt.subplot(k,6,i*6+j+1)
        plt.hist(list(dict_groceries[i][j]), bins=20, label=labels[j])
        
plt.savefig('all_clusters_groceries.png')
plt.clf()
