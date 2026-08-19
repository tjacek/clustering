import os,re

class NeuralModel(object):
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
     