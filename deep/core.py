import numpy as np
import os,re
import json
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
#    Dropout,
#    BatchNormalization,
    MaxPooling2D,
    UpSampling2D,
    Conv2DTranspose,
#    GlobalAveragePooling2D,
)

from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model 
from dataclasses import dataclass, asdict
import base
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
    
    def extract(self, data, n_layer=1):
        old_X = data.X if(isinstance(data, base.Dataset)) else data
        X = np.expand_dims(old_X.astype("float32") / 255.0, -1)
        extr=self.init_extractor(n_layer)
        feat = extr.predict(X, batch_size=256, verbose=0)
        return feat
    
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

@dataclass
class NNFactory:
    input_shape:tuple
#    n_cats:int
    dense_layers:list 
    n_kerns:list 
    kernel_sizes:list
    pool_size:list 

    def __post_init__(self):
        err1= "n_kerns and kernel_sizes must have the same length"
        assert len(self.n_kerns) == len(self.kernel_sizes),err1
        err2="pool_size musi mieć o jeden element mniej niż n_kerns"
        assert len(self.pool_size) == len(self.n_kerns) - 1,err2

    @property
    def n_conv(self):
        return len(self.n_kerns)
    
    @property
    def n_dense(self):
        return len(self.dense_layers)   
    
    def input_layer(self):
        return Input(shape=self.input_shape)     
    
    def get_pool(self,i):
        return MaxPooling2D(pool_size=self.pool_size[i])

    def get_conv(self,i):
        return conv_layer( i,
                           self.n_kerns[i],
                           self.kernel_sizes[i])
    def get_dense(self,i):
        return dense_layer( self.dense_layers[i],
                            f"layer_{i}")

def conv_layer( i,
                n_kerns,
                kernel_sizes):
    return Conv2D(filters=n_kerns,
                  kernel_size=kernel_sizes,
                  padding="same",
                  activation="relu",
                  name=f"enc_conv_{i}")

def deconv_layer( i,n_kerns, kernel_size):
    return Conv2DTranspose(
            filters=n_kerns,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name=f"dec_conv_{i}")

def dense_layer(dense,name):
    return Dense( dense,
                  activation="relu",
                  name=name)


@dataclass(frozen=True)
class _Hyperparams:
    input_shape:tuple
    n_cats:int
    dense_layers:list 
    n_kerns:list 
    kernel_sizes:list
    pool_size:list 
    

    
    @property
    def n_conv(self):
        return len(self.n_kerns)
    
    @property
    def n_dense(self):
        return len(self.dense_layers)   
    
    def input_layer(self):
        return Input(shape=self.input_shape)
    
   

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