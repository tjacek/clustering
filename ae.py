from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Dense,Dropout,Flatten
from tensorflow.keras.layers import BatchNormalization,MaxPooling2D,Reshape
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
import base

def make_ae(params):
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

def simple_exp(epochs=2,batch_size = 64,out_path=None):
    data=base.get_minst_dataset()
    params=default_params()
    autoencoder=make_ae(params=params)
    autoencoder.compile(optimizer="adam", 
                        loss="binary_crossentropy")
    autoencoder.summary()
    history = autoencoder.fit(data.x_train,
                              data.x_train,
                        batch_size=batch_size,
                        epochs=epochs)

    if(not out_path is None):
        autoencoder.save(out_path)
    predict=base.make_extractor(autoencoder,
                                "dense")#autoencoder.get_layer("dense")
    return base.Experiment(dataset=data,
                      model=autoencoder)
if __name__ == '__main__':
    simple_exp(out_path="simple_ann.h5")