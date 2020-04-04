"""
Queries arxiv API and downloads papers (the query is a parameter).
The script is intended to enrich an existing database pickle (by default db.p),
so this file will be loaded first, and then new results will be added to it.
"""

import os
import time
import pickle
import random
import argparse
import urllib.request
import feedparser

from utils import Config, safe_pickle_dump

if __name__ == "__main__":
  
  # lets load the existing database to memory
  try:
    db = pickle.load(open(Config.db_path, 'rb'))
  except Exception as e:
    print('error loading existing database:')
    print(e)
    print('starting from an empty database')
    db = {}

  # -----------------------------------------------------------------------------
  # main loop where we add txt files to db
  print('database has %d entries at start' % (len(db), ))
  num_added_total = 0
  text_files = set(os.listdir(Config.txt_dir))
  
  num_added = 0
  num_skipped = 0
  
  for i,f in enumerate(text_files):
 
    
    rawid = hash(str(f))
    
    # add to our database if we didn't have it before, or if this is a new version
    # we use the hash of filename as rawid
    if not rawid in db:
      db[rawid] = f
      print('Updated %s added' % f.encode('utf-8'))
      num_added += 1
      num_added_total += 1
    else:
      num_skipped += 1

  # print some information
  print('Added %d papers, skipped %d.' % (num_added, num_skipped))
  
  if num_added == 0:
    print('No new papers were added. Assuming no new papers exist. Exiting.')
    

  # save the database before we quit, if we found anything new
  if num_added_total > 0:
    print('Saving database with %d papers to %s' % (len(db), Config.db_path))
    safe_pickle_dump(db, Config.db_path)

