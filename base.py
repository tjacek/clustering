import tensorflow as tf
from tensorflow.keras import Input, Model
import numpy as np
from tensorflow.keras.models import load_model 

class Experiment(object):
    def __init__(self,dataset,model):
         self.dataset=dataset
         self.model=model
    
    def get_features(self,name_i='dense_1',batch_size=1024):
        ext=make_extractor(self.model,name_i)
        def helper(x):
            return ext.predict(x,batch_size=batch_size)
        return self.dataset.transform(helper)

    def all_names(self):
        return [layer.name for layer in self.model.layers]

def make_extractor(model,name_i='dense_1'):
    output= model.get_layer(name_i).output 
    return Model(inputs=model.input,
                 outputs=output)

class Dataset(object):
    def __init__(self,x_train, y_train,x_test, y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test

    def transform(self,fun):
    	return Dataset(x_train=fun(self.x_train), 
    		           y_train=self.y_train,
    		           x_test=fun(self.x_test), 
    		           y_test=self.y_test)

    def dim(self):
    	return self.x_train.shape

    def n_cats(self):
        return max(self.y_train)

    def get_cat(self,i):
        indices=(self.y_train==i)#[:,i]==1)
        x_i=self.x_train[indices]
        return x_i

    def save(self,out_path):
        args={'file':out_path,
              'x_train':self.x_train,
              'y_train':self.y_train,
              'x_test':self.x_test,
              'y_test':self.y_test}
        np.savez_compressed(**args)

def read_dataset(in_path):
    raw_data= np.load(in_path)
    return Dataset(x_train=raw_data['x_train'], 
                   y_train=raw_data['y_train'],
                   x_test=raw_data['x_test'], 
                   y_test=raw_data['y_test'])


def get_minst_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return Dataset(x_train, y_train,x_test, y_test)


def read_exp(in_path,
	         read_dataset=None):
    if(read_dataset is None):
        read_dataset=get_minst_dataset	
    model=load_model(in_path)
    return Experiment(dataset=read_dataset(),
    	              model=model)