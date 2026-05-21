import numpy as np
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Dense,Dropout,Flatten
from tensorflow.keras.layers import BatchNormalization,MaxPooling2D,Reshape
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
import cv2
import os,os.path
import base

class AE(object):
    def __init__(self,model):
        self.model=model
    
    def fit( self,
             data,
             epochs=50,
             batch_size = 64):
        self.model.compile( optimizer="adam", 
                            loss="mean_squared_error")
        X=np.expand_dims(data.X,-1)
        self.model.fit(X,X,
                       batch_size=batch_size,
                       epochs=epochs)
    
    def extract(self,data):#,n_layer=1):
        layer=self.model.get_layer(f"dense")
        extractor = Model( inputs=self.model.inputs,
                           outputs=layer.output)
        feat=extractor.predict(data.X)
        return feat

def make_ae(params=None):
    if(params is None):
        params=default_params()
    model = Sequential()
    model.add(Conv2D( params['n_kern1'], 
                      params['kern_size1'], 
                      activation="relu", 
                      padding="same",
                      input_shape=(28,28,1)))
    model.add(MaxPooling2D(params['max_pool1'], 
                           padding="same"))
    model.add(Conv2D( params['n_kern2'], 
                      params['kern_size2'], 
                     activation="relu", 
                     padding="same"))
    model.add(MaxPooling2D(params['max_pool2'], 
                           padding="same"))
    model.add(Flatten())  
    model.add(Dense(1568, activation='relu',name="dense"))
    model.add(Reshape(target_shape=(7, 7, 32)))
    model.add(Conv2DTranspose(params['n_kern2'], 
                              params['kern_size2'], 
                             strides=2, 
                             activation="relu", 
                             padding="same"))
    model.add(Conv2DTranspose(params['n_kern1'], 
                              params['kern_size1'], 
                             strides=2, 
                             activation="relu", 
                             padding="same"))
    model.add(Conv2D(1, 
                    (3, 3), 
                    activation="sigmoid",
                    padding="same"))
    return AE(model)


def default_params():
    return {'n_kern1':32, "kern_size1":(3,3),
            'n_kern2':32, "kern_size2":(3,3),
            'max_pool1':(2,2),'max_pool2':(2,2) }

def simple_exp(data=None,
               epochs=2,
               batch_size = 1024,
               out_path=None):
    if(data is None):
        data=base.get_minst_dataset()
    params=default_params()
    params['batch_size']=batch_size
    params['epochs']=epochs
    autoencoder= train_ae(data,params)
    if(not out_path is None):
        autoencoder.save(out_path)
    predict=base.make_extractor(autoencoder,
                                "dense")
    extractor=base.Experiment(dataset=data,
                      model=predict)
    return extractor,autoencoder

def train_ae(data,params):
    params['input_shape']=data.dim()
    print(params)
    autoencoder=make_ae(params=params)
    autoencoder.compile(optimizer="adam", 
                        loss="mean_squared_error")
    autoencoder.summary()
    history = autoencoder.fit(data.x_train,
                              data.x_train,
                        batch_size=params['batch_size'],
                        epochs=params["epochs"])
    return autoencoder

def save_imgs(data,autoencoder,out_path):
    print(data.x_train.shape)
    if(not os.path.exists(out_path)): 
        os.mkdir(out_path)
    for i,org_i in enumerate(data.x_train):
        org_i=np.expand_dims(org_i,axis=0)
        print(org_i.shape)
        rec_i=autoencoder.predict(org_i)
        org_i=np.squeeze(org_i)
        diff_i=np.abs(org_i-rec_i)
        
        out_i=f"{out_path}/{i}.png"
        cv2.imwrite(out_i,diff_i)

if __name__ == '__main__':
    data=base.get_minst_dataset()
    extractor,autoencoder=simple_exp(out_path="simple_ann.h5")
    save_imgs(data,autoencoder,"test")