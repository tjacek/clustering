import base
from dataclasses import dataclass,asdict,field
from itertools import product
import ae,cnn,cluster

@dataclass(frozen=True)
class Params:
    clustering:str = "spectral"
    model:str = "cnn"
    n_clusters:int = 2

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
                    n_clusters=self.n_clusters)
        return clust
    
    def __str__(self):
        return f"{self.model},{self.clustering},{self.n_clusters}"

@dataclass
class ParamsSpace:
    models: list = set_default(["cnn"])
    clust_algs: list = set_default(["kmeans"])
    n_clusters: list = set_default( [2, 5, 10])
    def __call__(self):
        as_dict=self.__dict__
        keys=list(as_dict.keys())
        keys.sort()
        values=[ as_dict[key_i] for key_i in keys]
        for params_i in product(*values):
            yield Params(*params_i) 

def set_default(value):
    return field(default_factory=lambda: value)

def simple_exp(*args):
    data=base.get_minst_dataset()
    params=Params(*args)
    clust=params(data)   
    clust.mean_img("test")

def multi_exp():
    param_space=ParamsSpace()
    print(param_space.models)
    for param_i in param_space():
        print(param_i)

#simple_exp( "cnn", "kmeans")
multi_exp()