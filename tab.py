import pandas as pd
import numpy as np
from sklearn import preprocessing
import gudhi as gd
import matplotlib.pyplot as plt

def read_csv(in_path:str):
    df=pd.read_csv(in_path)
    raw=df.to_numpy()
    X,y=raw[:,:-1],raw[:,-1]
    X= preprocessing.RobustScaler().fit_transform(X)
    print(X.shape)
    return X,y
#    return Dataset(X,y)

X,y=read_csv("../uci/cleveland")

rips_complex = gd.RipsComplex(points=X, 
                              max_edge_length=2.0)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=13)
diag=simplex_tree.persistence(min_persistence=0.4)
gd.plot_persistence_barcode(diag)
#ax = gd.plot_persistence_diagram(diag)
#ax.set_aspect("equal")  # 
plt.show()