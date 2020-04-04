import pickle
from utils import Config

# read database
db = pickle.load(open(Config.db_path, 'rb'))

for pid,j in db.items():
  print(pid, j)
  
print("Number of elements in db: " + str(len(db.items())))