import pandas as pd
import numpy as np
from sklearn import preprocessing
import gudhi as gd
import matplotlib.pyplot as plt
import utils

class Points(object):
    def __init__(self,X,y):
        self.X=X
        self.y=y

    def n_cats(self):
        return int(max(self.y)+1)
    
    def n_dims(self):
        return self.X.shape[1]

    def select(self,cat_i:int):
        return self.X[self.y==cat_i]

    def get_diag(self,cat_i=None):
        if(cat_i is None):
            points=self.X
        else:
            points=self.select(cat_i)
        rips_complex = gd.RipsComplex(points=points, 
                                      max_edge_length=1.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.n_dims())
        return simplex_tree.persistence(min_persistence=0.4)

def all_classes(in_path="../uci/cleveland",out_path=None):
    if(out_path is None):
        out_path=in_path.split('/')[-1]
    points=read_csv(in_path)
    utils.make_dir(out_path)
    for i in range(points.n_cats()):
        diag=points.get_diag(i)
        gd.plot_persistence_barcode(diag)
        plt.savefig(f'{out_path}/{i}')
        print(i)

def read_csv(in_path:str):
    df=pd.read_csv(in_path)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    print(X.shape)
    return Points(X,y)


all_classes(in_path="../uci/cmc")

#ax = gd.plot_persistence_diagram(diag)
#ax.set_aspect("equal")  # 
#plt.show()