import tensorflow as tf
from tensorflow.keras import Input, Model
import numpy as np
from tensorflow.keras.models import load_model 

class Dataset(object):
    def __init__(self,X,y):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.y)

    def dim(self):
        return self.X.shape

    def n_cats(self):
        return max(self.y)+1

    def cat_index(self,i):
        bool_values= (self.y==i)
        return [ i for i,bool_i in enumerate(bool_values)
                   if(bool_i)]

    def select(self,indexes):
        new_X,new_y=[],[]
        for i in indexes:
            new_X.append(self.X[i])
            new_y.append(self.y[i])
        return Dataset(X=np.array(new_X),
                       y=np.array(new_y))
    
    def __call__(self,fun):
        new_X=[fun(x_i) for x_i in self.X]
        return Dataset(X=np.array(new_X),
                       y=self.y)

    def subsample(self,p=0.1):
        new_X=[]
        for x_i in self.X:
            if(p < np.random.uniform()):
                new_X.append(x_i)
        return Dataset(X=np.array(new_X),
                       y=self.y)

class DataPair(object):
    def __init__( self,
                  train,
                  test):
        self.train=train
        self.test=test

class Split(object):
    def __init__(self,train_index,test_index):
        self.train_index=train_index
        self.test_index=test_index

    def eval(self,data,clf):
        return data.eval(train_index=self.train_index,
                         test_index=self.test_index,
                         clf=clf)


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

def read_dataset(in_path):
    raw_data= np.load(in_path)
    return Dataset(x_train=raw_data['x_train'], 
                   y_train=raw_data['y_train'],
                   x_test=raw_data['x_test'], 
                   y_test=raw_data['y_test'])


def get_minst_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train = Dataset(x_train,y_train)
    test = Dataset(x_test,y_test)
    return DataPair(train,test)

def read_exp(in_path,
	         read_dataset=None):
    if(read_dataset is None):
        read_dataset=get_minst_dataset	
    model=load_model(in_path)
    return Experiment(dataset=read_dataset(),
    	              model=model)