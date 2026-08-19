import os,re
import json
from dataclasses import dataclass, asdict

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
        self._extractor = None
        self.extractor_layer = None
        
    @classmethod
    def get_model(cls,out_path,data=None):
        if(os.path.exists(out_path)):
            return cls.read(out_path)
        if(data is None):
            data=get_minst_dataset()
        if(type(data)==DataPair):
            data=data.train
        nn_model=cls.make()
        nn_model.fit(data)
        nn_model.save(out_path)
        return nn_model
     
    @classmethod
    def read(cls,in_path):
        model = load_model(in_path)
        return cls(model)

    def save(self,out_path):
        if(len(out_path.split("."))<2):
            out_path+=".keras"
        self.model.save(out_path)

    def names(self):
        return [ layer_i.name 
                for layer_i in self.model.layers]

    def find_layers(self,regex=r'^layer_\d+'):
        return  [name_i 
                    for name_i in self.names()
                        if( re.match(regex,name_i))]
    

#class NNFactory(object):
