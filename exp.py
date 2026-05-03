import base
from dataclasses import dataclass
import ae,cnn

@dataclass(frozen=True)
class Params:
    model:str = "cnn"
    clustering:str = "kmeans"

    def make_model(self):
    	if(self.model=="cnn"):
    		return cnn.make_cnn()
       if(self.model=="ae"):
    		return ae.make_ae()

def simple_exp(*args):
	params=base.Params(*args)
	print(params)

simple_exp( "cnn", "kmeans")
