import base
from dataclasses import dataclass
import ae,cnn,cluster

@dataclass(frozen=True)
class Params:
    model:str = "cnn"
    clustering:str = "kmeans"
    
    def make_model(self):
        if(self.model=="cnn"):
            model=cnn.make_cnn()
        if(self.model=="ae"):
            model=ae.make_ae()
        return model

    def clust_alg(self):
        if(self.clustering=="kmeans"):
            return cluster.kmeans_alg

def simple_exp(*args):
    data=base.get_minst_dataset()
    params=Params(*args)
    model=params.make_model()
    model.fit(data.train)
    feat=model.extract(data.train)
    clust_alg=params.clust_alg()
    clust=clust_alg(data,
                    feat,
                    n_clusters=2)
    clust.mean_img("test")

simple_exp( "cnn", "kmeans")
