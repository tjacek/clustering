import tensorflow as tf
from tensorflow.keras import Input, Model
import numpy as np
from tensorflow.keras.models import load_model 

class Experiment(object):
    def __init__(self,dataset,model):
         self.dataset=dataset
         self.model=model
    
    def get_features(self,name_i='dense_1',batch_size=1024):
        ext=self.make_extractor(name_i)
        def helper(x):
            return ext.predict(x,batch_size=batch_size)
        return self.dataset.transform(helper)

    def all_names(self):
        return [layer.name for layer in self.model.layers]

    def make_extractor(self,name_i='dense_1'):
        output= self.model.get_layer(name_i).output 
        return Model(inputs=self.model.input,
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

def centroid_distance(dataset):
    groups=[ dataset.get_cat(i)
             for i in range(dataset.n_cats())]
    centroids=[ np.mean(group_i,axis=0)  
               for group_i in groups]
    distances=[[np.linalg.norm(centroid_i-point_j) 
                    for point_j in groups[i]]
                for i,centroid_i in enumerate(centroids)]
    dist_i=distances[0]
    dist_i.sort()
    print(dist_i)
    return distances  

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

def simple_exp(epochs=50,batch_size = 64,out_path=None):
    data=get_minst_dataset()
    params=default_params()
    model=make_cnn(params)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), 
    	          loss='categorical_crossentropy', 
    	          metrics=['acc'])
    callbacks=SimpleCallback()
    history = model.fit(data.x_train, data.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        callbacks=[callbacks])
    if(not out_path is None):
        model.save(out_path)
    return Experiment(dataset=data,
    	              model=model)


#exp=simple_exp(out_path="simple_cnn")
exp=read_exp("simple_cnn")
feat=exp.get_features('dense')
centroid_distance(feat)
#print(feat.get_cat(1).shape)