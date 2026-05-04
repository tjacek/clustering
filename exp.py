import base
from dataclasses import dataclass
import ae,cnn

@dataclass()#frozen=True)
class Params:
    model:str = "cnn"
    clustering:str = "kmeans"
    dataset= None
    
    def make_model(self):
        data=self.get_data().train
        if(self.model=="cnn"):
            model=cnn.make_cnn()
#            model.fit(data.X,
#                      data.y)
        if(self.model=="ae"):
            model=ae.make_ae()
        model.fit(data)
        return model

    def get_data(self):
        if(self.dataset is None):
            self.dataset=base.get_minst_dataset()
        return self.dataset

#    def make_model(self):
#        if(self.model=="cnn"):
#    		   model=cnn.make_cnn()
#            model.fit(dataset.X,
#                      dataset.y)
#        if(self.model=="ae"):
#    	      model=ae.make_ae()
#            model.fit(dataset.X)
#        return model

def simple_exp(*args):
   params=Params(*args)
   params.make_model()
   print(params)

simple_exp( "cnn", "kmeans")
