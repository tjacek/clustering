import os,re
import json
from dataclasses import dataclass, asdict
import utils
@dataclass(frozen=True)
class NNMeta:
    nn_type: str
    builder_type: str
    hyper_params: dict

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
    def __init__( self,
                  model,
                  nn_meta ):
        self.model=model
        self.nn_meta=nn_meta
        self.extractor = None
        self.extractor_layer = None

    def save(self,out_path):
        utils.make_dir(out_path)
        self.nn_meta.save(f"{out_path}/meta")
        self.model.save(f"{out_path}/model.keras")
    
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
        model = load_model(in_path)
        return cls(model)