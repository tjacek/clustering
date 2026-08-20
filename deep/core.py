import os,re
import json
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
#    Dropout,
#    BatchNormalization,
    MaxPooling2D,
    UpSampling2D,
#    GlobalAveragePooling2D,
)

from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model 
from dataclasses import dataclass, asdict
import utils

@dataclass#(frozen=True)
class NNMeta:
    nn_type: str
    builder_type: str
    hyper_params: dict
    n_epochs: int = 0

    def save(self, out_path: str):
        if not out_path.endswith(".json"):
            out_path += ".json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def read(cls, in_path: str):
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

class NeuralModel(object):
    MODEL_FILE="model.keras"
    META_FILE="meta.json"
    def __init__( self,
                  model,
                  nn_meta ):
        self.model=model
        self.nn_meta=nn_meta
        self.extractor = None
        self.extractor_layer = None

    def save(self,out_path):
        utils.make_dir(out_path)
        self.nn_meta.save(f"{out_path}/{self.META_FILE}")
        self.model.save(f"{out_path}/{self.MODEL_FILE}")
    
    def init_extractor(self,n_layer=0):   
        if(self.extractor is None or 
              self.extractor_layer!=n_layer):
            layer = self.model.get_layer(f"layer_{n_layer}")
            self.extractor = Model(
                                inputs=self.model.inputs,
                                outputs=layer.output,
                              )
            self.n_layer=n_layer
        return self.extractor
    
    def names(self):
        return [ layer_i.name 
                for layer_i in self.model.layers]

    def find_layers(self,regex=r'^layer_\d+'):
        return  [name_i 
                    for name_i in self.names()
                        if( re.match(regex,name_i))]

    @classmethod
    def read(cls,in_path):
        nn_meta=NNMeta.read(f"{in_path}/{cls.META_FILE}")
        model = load_model(f"{in_path}/{cls.MODEL_FILE}")
        return cls(model,nn_meta)


@dataclass(frozen=True)
class Hyperparams:
    input_shape:tuple
    n_cats:int
    dense_layers:list 
    n_kerns:list 
    kernel_sizes:list
    pool_size:list 
    
    def __post_init__(self):
        assert len(self.n_kerns) == len(self.kernel_sizes), "n_kerns and kernel_sizes must have the same length"
        assert len(self.pool_size) == len(self.n_kerns) - 1, "pool_size musi mieć o jeden element mniej niż n_kerns"
    
    @property
    def n_conv(self):
        return len(self.n_kerns)
    
    @property
    def n_dense(self):
        return len(self.dense_layers)   
    
    def input_layer(self):
        return Input(shape=self.input_shape)
    
    def conv_layer(self,i):
        args={ "filters":self.n_kerns[i],
               "kernel_size":self.kernel_sizes[i],
               "padding":"same",
               "activation":"relu"}
#        if(i==0):
#            args["input_shape"]=self.input_shape
        return Conv2D(**args)

    def dense_layer(self,i):
        return Dense( self.dense_layers[i],
                      activation="relu",
                      name=f"layer_{i}")

    def pool_layer(self,i):
        return MaxPooling2D(pool_size=self.pool_size[i])
    
    def rev_kerns(self):
        return list(reversed(self.n_kerns))

    def rev_sizes(self):
        return list(reversed(self.kernel_sizes))

    def rev_pool_indices(self):
        return list(reversed(range(len(self.pool_size)))) + [None]
    
    def upsample_layer(self, i):
        return UpSampling2D(size=self.pool_size[i])

def minst_params(n_cats=10):
    return Hyperparams(
            input_shape=(28,28,1),
            n_cats=n_cats,
            dense_layers=[1024,512],
            n_kerns=[32,32,64],
            kernel_sizes=[(3,3),(3,3),(3,3)],
            pool_size=[(2,2),(2,2)]
        )

def frame_params(n_cats=20):
    return Hyperparams(
            input_shape=(240, 80, 1),
            n_cats=n_cats,
            dense_layers=[1024,128],
            n_kerns=[32,64,128,256],
            kernel_sizes=[(5, 3),(3,3),(3,3),(3,3)],
            pool_size=[(2, 2),(2, 2),(2, 2)]
        )