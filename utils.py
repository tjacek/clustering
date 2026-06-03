import os.path
import itertools

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def index_pairs(arr):
    indices=range(len(arr))
    for i,j in itertools.combinations(indices,2):
        yield i,j