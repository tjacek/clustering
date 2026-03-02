import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,BatchNormalization,MaxPooling2D
from tensorflow.keras import Input, Model
from sklearn.metrics import accuracy_score
import base 

class ConvNN(object):
    def __init__(self,model):
        self.model=model

    def fit( self,
             data,
             epochs=50,
             batch_size = 64):
        optim=tf.keras.optimizers.RMSprop(epsilon=1e-08)
        self.model.compile( optimizer=optim, 
                            loss='categorical_crossentropy', 
                            metrics=['acc'])
        callbacks=SimpleCallback()
        X=np.expand_dims(data.X,-1)
        y=tf.one_hot(data.y,
                     depth=10)
        self.model.fit(X,y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.1,
                       callbacks=[callbacks])

    def eval( self,
              data):
        y=self.model.predict(data.X)
        y=np.argmax(y,axis=1)
        acc=accuracy_score(data.y,y)
        return acc
    
    def extract(self,data,n_layer=1):
        layer=self.model.get_layer(f"layer_{n_layer}")
        extractor = Model( inputs=self.model.inputs,
                           outputs=layer.output)
        feat=extractor.predict(data.X)
        return feat

class SimpleCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
        self.model.stop_training = True


def make_cnn(params):
    model = Sequential()

    model.add(Conv2D(filters=params['n_kern1'], 
                     kernel_size=params['kern_size1'], 
                     activation='relu', strides=1, 
                     padding='same', 
                     data_format='channels_last',
                      input_shape=(28,28,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=params['n_kern2'], 
                     kernel_size=params['kern_size2'],
                     activation='relu', 
                     strides=1, padding='same', 
                     data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=2, 
                           padding='valid' ))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=params['n_kern3'], 
                     kernel_size=params['kern_size3'],
                     activation='relu', 
                     strides=1, 
                     padding='same', 
                     data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Conv2D( filters=64, 
                      kernel_size=(3, 3), 
                      strides=1, 
                      padding='same', 
                      activation='relu', 
                      data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(params['n_layer1'], 
                    activation='relu',
                    name="layer_2"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense( params['n_layer2'], 
                     activation='relu',
                     name="layer_1"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(params['n_cats'], activation='softmax'))
    model.summary()
    return ConvNN(model)

def default_params(n_layer1=1024,
                   n_layer2=512):
    return {'n_layer1':n_layer1,
            'n_layer2':n_layer2,
            'n_kern1':32, "kern_size1":(3,3),
            'n_kern2':32, "kern_size2":(3,3),
            'n_kern3':64, "kern_size3":(3,3),  
            "n_cats":10}

def simple_exp(data=None,
               n_neurons=512):
    if(data is None):
        data=base.get_minst_dataset()
    params=default_params(n_layer1=n_neurons)
    model=make_cnn(params)
    model.fit(data.train)
    acc=model.eval(data.test)
    print(f"{acc:.4f}")
    return model,data

if __name__ == '__main__':
    model,data=simple_exp()
    model.extract(data.train)