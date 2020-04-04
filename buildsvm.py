# standard imports
import os
import sys
import pickle
# non-standard imports
import numpy as np
from sklearn import svm
from sqlite3 import dbapi2 as sqlite3
# local imports
from utils import safe_pickle_dump, strip_version, Config

num_recommendations = 1000 # papers to recommend per user


# load the tfidf matrix and meta
meta = pickle.load(open(Config.meta_path, 'rb'))
out = pickle.load(open(Config.tfidf_path, 'rb'))
X = out['X']
X = X.todense().astype(np.float32)

xtoi = { strip_version(x):i for x,i in meta['ptoi'].items() }


print("building an SVM ")
lib = query_db('''select * from library where user_id = ?''', [uid])
pids = [x['paper_id'] for x in lib] # raw pids without version
posix = [xtoi[p] for p in pids if p in xtoi]

if not posix:
  continue # empty library for this user maybe?

print(pids)
y = np.zeros(X.shape[0])
for ix in posix: y[ix] = 1

clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
clf.fit(X,y)
s = clf.decision_function(X)

sortix = np.argsort(-s)
sortix = sortix[:min(num_recommendations, len(sortix))] # crop paper recommendations to save space
user_sim = [strip_version(meta['pids'][ix]) for ix in list(sortix)]

print('writing', Config.user_sim_path)
safe_pickle_dump(user_sim, Config.user_sim_path)
