import os.path,re
import itertools

def top_files(path):
    paths=[ path_i for id_i,path_i in iter_files(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def sort(items):
    return sorted(items,key=natural_keys)

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

def print_dict(d):
    for key_i,value_i in d.items():
        print(key_i)
        print(value_i)

def find_paths(in_path,regex=r'^layer_\d+'):
    paths= [path_i 
                for id_i,path_i in iter_files(in_path)
                    if( re.match(regex,id_i))]
    paths=sorted(paths,key=natural_keys)
    return paths