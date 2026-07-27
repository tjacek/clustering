import os.path
import itertools

def iter_files(path):
    if(type(path)==str):
         path=[path]
    for dir_i in path:
        for file_i in os.listdir(dir_i):
            yield  file_i,f'{dir_i}/{file_i}'

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def index_pairs(arr):
    indices=range(len(arr))
    for i,j in itertools.combinations(indices,2):
        yield i,j

def pair_iter(x,y):
    x=range(len(x))
    y=range(len(y))
    for i in x:
        for j in y[i:]:
            yield i,j