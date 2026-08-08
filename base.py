import tensorflow as tf
from tensorflow.keras import Input, Model
import numpy as np
from tensorflow.keras.models import load_model 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os,re
import cv2
import utils 

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
        new_X,new_y=[],[]
        for i,x_i in enumerate(self.X):
            if( np.random.uniform() < p):
                new_X.append(x_i)
                new_y.append(self.y[i])
        return Dataset(X=np.array(new_X),
                       y=np.array(new_y))

    def save(self,out_path):
        utils.make_dir(out_path)
        for i,x_i in enumerate(self.X):
            out_ij=f"{out_path}/{i}.png"
            cv2.imwrite(out_ij,x_i)

class DataPair(object):
    def __init__( self,
                  train,
                  test):
        self.train=train
        self.test=test

    def subsample(self,p=0.1):
        return DataPair(self.train.subsample(p),
                        self.test.subsample(p))

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
     
class Features(np.ndarray):
    def __new__( cls,
                 input_array, 
                 info=None,
                 feat_type="base"):
        obj = np.asarray(input_array).view(cls)
        if(type(info)==Dataset):
            info={"data":info}
        obj.info=info
        obj.feat_type=feat_type
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)
        self.feat_type = getattr(obj, 'feat_type', None)
    
    def __call__(self,name):
        if(name in self.info):
            return self.info[name]
        if(name=="X"):
            return self.info["data"].X 
        if(name=="y"):
            return self.info["data"].y

    def select(self,indices):
        X=self("X")[indices]
        y=self("y")[indices]
        feat=self[indices]
        data=Dataset(X,y)
        return Features(feat,data)

    def to_pca(self):
        pca = PCA(n_components=None)
        feats=pca.fit_transform(self)
        info=self.info.copy()
        info["var"]=pca.explained_variance_
        return PcaFeatures( feats,info,"PCA")

    def n_cats(self):
        return self.info["data"].n_cats()

    def mean_norm(self):
        dist=0
        for x_i in self:
            dist+=l0_norm(x_i)
        return dist/len(self)

class PcaFeatures(Features):
    def cum_var(self):
        var=self.info["var"]
        return np.cumsum(var/np.sum(var))

    def plot(self,i,j):
        x,y=self[:,i],self[:,j]
        labels=self.info["data"].y
        fig, ax = plt.subplots()
        colors=["b",'g','r','c','m',"y",
                "orange",'pink','gray','brown']
        point_colors=[colors[i % len(colors)]  
                         for i in labels]
        ax.scatter(x, y,c=point_colors)
        for i, txt in enumerate(labels):
            
            ax.annotate(str(txt), (x[i], y[i]))
        plt.show()

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

def get_metric(metric_type="L2",arr=False):
    if(metric_type==None):
        return L2
    if(metric_type=="L2" and not arr):
        return L2
    if(metric_type=="L2" and arr):
        return L2_arr
    if(metric_type=="L1" and not arr):
        return L1
    if(metric_type=="L1" and arr):
        return L1_arr
    if(metric_type=="cos" and not arr):
        return cos_metric
    if(metric_type=="cos" and arr):
        return cos_arr

def L2(a,b):
    return np.linalg.norm(a-b)

def L1(a,b):
    return np.sum(np.abs(a-b))

def cos_metric(a,b):
    a_len=np.linalg.norm(a)
    b_len=np.linalg.norm(b)
    return 1.0 - np.dot(a,b)/(a_len*b_len)

def L2_arr(x,b):
    diff=x-b
    diff=diff*diff
    return np.sqrt(np.sum(diff,axis=1))

def L1_arr(x,b):
    diff=np.abs(x-b)
    return np.sqrt(np.sum(diff,axis=1))

def cos_arr(x,b):
    x_len=l2_norm_arr(x)
    b_len=np.linalg.norm(b)
    return 1.0 - np.dot(x,b)/(x_len*b_len)

def l2_norm_arr(x):
    return np.sqrt(np.sum(x*x,axis=1))


def l0_norm(x):
    return np.sum((x==0).astype(int))