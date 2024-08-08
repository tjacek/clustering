from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Dense,Dropout,Flatten
from tensorflow.keras.layers import BatchNormalization,MaxPooling2D,Reshape
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
import cv2
import base

def make_ae(params):
    model = Sequential()
    model.add(Conv2D( params['n_kern1'], 
                      params['kern_size1'], 
                      activation="relu", 
                      padding="same",
                      input_shape=params['input_shape']))
    model.add(MaxPooling2D(params['max_pool1'], 
                           padding="same"))
    model.add(Conv2D( params['n_kern2'], 
                      params['kern_size2'], 
                     activation="relu", 
                     padding="same"))
    model.add(MaxPooling2D(params['max_pool2'], 
                           padding="same"))
    model.add(Flatten())  
    model.add(Dense(1568, activation='relu'))
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
    return model


def default_params():
    return {'n_kern1':32, "kern_size1":(3,3),
            'n_kern2':32, "kern_size2":(3,3),
            'max_pool1':(2,2),'max_pool2':(2,2) }

def simple_exp(data=None,
               epochs=2,
               batch_size = 64,
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

    autoencoder=make_ae(params=params)
    autoencoder.compile(optimizer="adam", 
                        loss="mean_squared_error")
    autoencoder.summary()
    history = autoencoder.fit(data.x_train,
                              data.x_train,
                        batch_size=params['batch_size'],
                        epochs=epochs)
    return autoencoder

def save_imgs(data,autoencoder,out_path):
    print(data.x_train.shape)
    for i,x_i in enumerate(data.x_train):
        out_i=f"{out_path}/{i}"
        cv2.imwrite(out_i,x_i)

if __name__ == '__main__':
    data=base.get_minst_dataset()
    extractor,autoencoder=simple_exp(out_path="simple_ann.h5")
    save_imgs(data,autoencoder,"test")