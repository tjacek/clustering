from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Dense,Dropout,Flatten,BatchNormalization,MaxPooling2D
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
import base

def make_ae(params):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), 
                      activation="relu", 
                      padding="same",
                      input_shape=(28,28,1)))
    model.add(MaxPooling2D((2, 2), 
                           padding="same"))
    model.add(Conv2D(32, 
                     (3, 3), 
                     activation="relu", 
                     padding="same"))
    model.add(MaxPooling2D((2, 2), 
                           padding="same"))
#    model.add(Flatten())  
#    model.add(Dense(512, activation='relu'))
    model.add(Conv2DTranspose(32, 
                             (3, 3), 
                             strides=2, 
                             activation="relu", 
                             padding="same"))
    model.add(Conv2DTranspose(32, 
                             (3, 3), 
                             strides=2, 
                             activation="relu", 
                             padding="same"))
    model.add(Conv2D(1, 
                    (3, 3), 
                    activation="sigmoid",
                    padding="same"))
    return model

def simple_exp(epochs=2,batch_size = 64,out_path=None):
    data=base.get_minst_dataset()
    autoencoder=make_ae(params=None)
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
                                "max_pooling2d_1")#autoencoder.get_layer("dense")
    return base.Experiment(dataset=data,
                      model=autoencoder)
if __name__ == '__main__':
    simple_exp(out_path="simple_ann.h5")