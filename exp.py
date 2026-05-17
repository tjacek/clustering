import base
from dataclasses import dataclass
import ae,cnn,cluster

@dataclass(frozen=True)
class Params:
    model:str = "cnn"
    clustering:str = "spectral"
    
    def make_model(self):
        if(self.model=="cnn"):
            model=cnn.make_cnn()
        if(self.model=="ae"):
            model=ae.make_ae()
        return model

    def clust_alg(self):
        if(self.clustering=="kmeans"):
            return cluster.kmeans_alg
        elif(self.clustering=="spectral"):
            return cluster.spectral_alg

    def __call__(self,data):
        model=self.make_model()
        model.fit(data.train)
        feat=model.extract(data.train)
        clust_alg=self.clust_alg()
        clust=clust_alg(data,
                    feat,
                    n_clusters=2)
        return clust

class ParamsSpace(object):
    def __init__(self,models, clust_algs):
        self.models=models
        self.clust_algs=clust_algs

    def __call__(self):
        for model_i in self.models:
            for clust_j in self.clust_algs:
                yield Params(model_i,clust_j) 

def simple_exp(*args):
    data=base.get_minst_dataset()
    params=Params(*args)
    clust=params(data)   
    clust.mean_img("test")

simple_exp( "cnn", "kmeans")
