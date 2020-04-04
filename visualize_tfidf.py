# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 12:59:37 2020

@author: fulop

Visualizing papers using TF-IDF + PCA, t-SNE

Loads the data from the tfidf.p, created by analyze.py

some code from
https://stackoverflow.com/questions/27494202/how-do-i-visualize-data-points-of-tf-idf-vectors-for-kmeans-clustering

# https://stackoverflow.com/questions/7908636/possible-to-make-labels-appear-when-hovering-over-a-point-in-matplotlib

"""

#%% standard imports
import os
import sys
import pickle
# non-standard imports
import numpy as np

from utils import safe_pickle_dump, strip_version, Config


import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

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
    annot.get_bbox_patch().set_alpha(0.9)


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


#%% Load previously compiled data and run PCA, t-SNE


# load the tfidf matrix and meta
meta = pickle.load(open(Config.meta_path, 'rb'))

vocab = meta['vocab']

out = pickle.load(open(Config.tfidf_path, 'rb'))
X = out['X']
X = X.todense().astype(np.float32) # shape: (no. of documents, vocab size)

names = meta['fnames']

# vocab_inverse = {v: k for k, v in vocab.items()}
pca_num_components = 2
tsne_num_components = 2


reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
embeddings = TSNE(n_components=tsne_num_components,random_state=2020,learning_rate=200)

Y_tsne = embeddings.fit_transform(X)
#%% PCA and t-SNE scatterplot, coloring based on vocabulary/bigram (saves to file)

bigrams = ['quantum','josephson','qubit','machine learning','graphene',
           'nanowire','microwave', 'noise', 'quantum dot', 'electronics', 
           'quality factor', 'superconductor', 'epitaxial', 'physics',
           'topological','semiconductor','condensed matter','cryogenic',
           'engineering','neural']

# bigrams = ['quantum']

for bigram in bigrams:

  if bigram in vocab:
    bigram_id = vocab[bigram]
    print('id of %s : %d' % (bigram, bigram_id ))
  else:
    print('bigram not found in vocabulary: %s' % bigram)
    break
  
  threshold = 0.01
  # threshold = np.mean(X[:,bigram_id])
  #threshold = np.quantile(X[:,bigram_id],0.7)
  
  true_document_idx = np.asarray(X[:,bigram_id] >= threshold)
  true_document_idx = true_document_idx.reshape(-1).T
  
  cmap = cm.get_cmap('coolwarm')
  
  fig, ax = plt.subplots(1,2,figsize=(12,5))
  sc = ax[0].scatter(reduced_data[:,0], reduced_data[:,1],c=true_document_idx,cmap=cmap)

  # sc = ax[0].scatter(reduced_data[true_document_idx,0], reduced_data[true_document_idx,1],color='g', label = bigram)
  # sc = ax[0].scatter(reduced_data[~true_document_idx,0], reduced_data[~true_document_idx,1],color='b', label = "not "+ bigram)
  true_patch = mpatches.Patch(color=cmap(cmap.N), label=bigram)
  false_patch = mpatches.Patch(color=cmap(0), label="not " + bigram)

  ax[0].legend(handles=[true_patch, false_patch])
  
  ax[0].set_title("PCA, n-gram: %s" % bigram)
  ax[0].set_xlabel('Principal axis 1')
  ax[0].set_ylabel('Principal axis 2')
  


  sc = ax[1].scatter(Y_tsne[:,0], Y_tsne[:,1],c=true_document_idx,cmap=cmap)

  ax[1].legend(handles=[true_patch, false_patch])
  # sc = ax[1].scatter(Y_tsne[true_document_idx,0], Y_tsne[true_document_idx,1],color='g', label = bigram)
  # sc = ax[1].scatter(Y_tsne[~true_document_idx,0], Y_tsne[~true_document_idx,1],color='b', label = "not "+ bigram)
  ax[1].legend(handles=[true_patch, false_patch])
  
  ax[1].set_title("t-SNE, n-gram: %s" % bigram)

  ax[1].set_xlabel('Embedding axis 1')
  ax[1].set_ylabel('Embedding axis 2')
  
  plt.show()
  plt.savefig("plots/" + bigram + ".png")
  plt.close()

#%% interactive plot, shows filename when hovering (PCA)
bigram = 'neural'

if bigram in vocab:
  bigram_id = vocab[bigram]
  print('id of %s : %d' % (bigram, bigram_id ))
  
  threshold = np.mean(X[:,bigram_id])
  
  true_document_idx = np.asarray(X[:,bigram_id] >= threshold)
  
  cmap = cm.get_cmap('coolwarm')
 
  
  fig, ax = plt.subplots()
  sc = ax.scatter(reduced_data[:,0], reduced_data[:,1],c=true_document_idx,cmap=cmap)
  
  true_patch = mpatches.Patch(color=cmap(cmap.N), label=bigram)
  false_patch = mpatches.Patch(color=cmap(0), label="not " + bigram)

  ax.legend(handles=[true_patch, false_patch])
  
  annot = ax.annotate("", xy=(-200,0), xytext=(-200,20),textcoords="offset points",fontsize =9,
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"))
  annot.set_visible(False)
  
  fig.canvas.mpl_connect("motion_notify_event", hover)
  
  plt.title("PCA")
  plt.xlabel('Principal axis 1')
  plt.ylabel('Principal axis 2')
  plt.show()
else:
  print('bigram not found in vocabulary: %s' % bigram)
  
  

#%% interactive plot, shows filename when hovering (t-SNE)
bigram = 'neural'

if bigram in vocab:
  bigram_id = vocab[bigram]
  print('id of %s : %d' % (bigram, bigram_id ))
  
  threshold = np.mean(X[:,bigram_id])
  
  true_document_idx = np.asarray(X[:,bigram_id] >= threshold)
  
  cmap = cm.get_cmap('coolwarm')
 
  
  fig, ax = plt.subplots()
  sc = ax.scatter(Y_tsne[:,0], Y_tsne[:,1],c=true_document_idx,cmap=cmap)
  
  true_patch = mpatches.Patch(color=cmap(cmap.N), label=bigram)
  false_patch = mpatches.Patch(color=cmap(0), label="not " + bigram)

  ax.legend(handles=[true_patch, false_patch])
  
  annot = ax.annotate("", xy=(-200,0), xytext=(-200,20),textcoords="offset points",fontsize =9,
                      bbox=dict(boxstyle="round", fc="w"),
                      arrowprops=dict(arrowstyle="->"))
  annot.set_visible(False)
  
  fig.canvas.mpl_connect("motion_notify_event", hover)
  
  ax.set_title("t-SNE, n-gram: %s" % bigram)

  ax.set_xlabel('Embedding axis 1')
  ax.set_ylabel('Embedding axis 2')
  plt.show()
else:
  print('bigram not found in vocabulary: %s' % bigram)




