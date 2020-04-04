# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:59:37 2020

@author: fulop

Visualize the document-term frequency matrix with PCA and t-SNE

Loads the data from the tfidf.p, created by analyze.py

mostly from
https://stackoverflow.com/questions/27494202/how-do-i-visualize-data-points-of-tf-idf-vectors-for-kmeans-clustering

# https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib

"""

# standard imports
import os
import sys
import pickle
# non-standard imports
import numpy as np

from utils import safe_pickle_dump, strip_version, Config


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos[0] , pos[1]
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([str(names[n]) for n in ind["ind"]]))
    annot.set_text(text)
    #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


#%%


# load the tfidf matrix and meta
meta = pickle.load(open(Config.meta_path, 'rb'))



out = pickle.load(open(Config.tfidf_path, 'rb'))
X = out['X']
X = X.todense().astype(np.float32) # shape: (no. of documents, vocab size)

names = meta['fnames']


pca_num_components = 2

reduced_data = PCA(n_components=pca_num_components).fit_transform(X)

#%% scatterplot, coloring based on vocabulary/bigram

vocab = meta['vocab']

bigrams = ['quantum','majorana','qubit','machine learning']

for bigram in bigrams:

  if bigram in vocab:
    bigram_id = vocab[bigram]
    print('id of %s : %d' % (bigram, bigram_id ))
  else:
    print('bigram not found in vocabulary: %s' % bigram)
    break
  
  #threshold = 0.01
  threshold = np.mean(X[:,bigram_id])
  #threshold = np.quantile(X[:,bigram_id],0.7)
  
  true_document_idx = np.asarray(X[:,bigram_id] >= threshold)
  true_document_idx = true_document_idx.reshape(-1).T
  
  
  fig, ax = plt.subplots()
  
  sc = ax.scatter(reduced_data[true_document_idx,0], reduced_data[true_document_idx,1],color='g', label = bigram)
  sc = ax.scatter(reduced_data[~true_document_idx,0], reduced_data[~true_document_idx,1],color='b', label = "not "+ bigram)
  
  plt.xlabel('Principal axis 1')
  plt.ylabel('Principal axis 2')
  
  plt.title("PCA, bigram: %s" % bigram)
  plt.legend()
  plt.show()
  plt.savefig("PCA_" + bigram + ".png")

#%% interactive plot, show filename when hovering

# print reduced_data

fig, ax = plt.subplots()
sc = ax.scatter(reduced_data[:,0], reduced_data[:,1])
plt.title("PCA")

annot = ax.annotate("", xy=(-200,0), xytext=(-200,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.xlabel('Principal axis 1')
plt.ylabel('Principal axis 2')
plt.show()
#%%

tsne_num_components = 2


# t-SNE plot
embeddings = TSNE(n_components=tsne_num_components)
Y = embeddings.fit_transform(X)

fig, ax = plt.subplots()
sc = plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
plt.title("t-SNE")

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()




