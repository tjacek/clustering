import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras import Model
from sklearn.metrics import accuracy_score
from dataclasses import dataclass
import base

class ConvNN(base.NeuralModel):
    def __init__(self,model):
        self.model=model
        self._extractor=None

    @classmethod
    def make(cls, params=None, verbose=False):
        return make_cnn(params, verbose)

    def fit(self, data, epochs=50, batch_size=64):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        callbacks = [SimpleCallback()]

        X = np.expand_dims(data.X.astype("float32") / 255.0, -1)

        self.model.fit( 
            X,
            data.y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
        )

    def eval(self, data):
        X = np.expand_dims(data.X.astype("float32") / 255.0, -1)

        y_pred = self.model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)

        return accuracy_score(data.y, y_pred)
    

    def extract(self, data, n_layer=1):
        old_X = data.X if(type(data)==base.Dataset) else data
        X = np.expand_dims(old_X.astype("float32") / 255.0, -1)

        layer = self.model.get_layer(f"layer_{n_layer}")

        if(self._extractor is None):
            self._extractor = Model(
                                inputs=self.model.inputs,
                                outputs=layer.output,
                              )

        feat = self._extractor.predict(X, batch_size=256, verbose=0)
        return feat

    def predict(self,X):
        prob= self.model.predict(X,verbose=0)
        return np.argmax(prob,axis=1)

class SimpleCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy")

        if acc is not None and acc > 0.995:
            print("\nReached 99.5% accuracy, stopping training.")
            self.model.stop_training = True

def make_cnn(params=None, verbose=False):
    if params is None:
        params = minst_params()

    model = Sequential()
    for n_kerns_i,sizes_i,pool_i in params:
        if(pool_i is None):
            model.add(Conv2D(
                          filters=n_kerns_i,
                          kernel_size=sizes_i,
                          padding="same",
                          activation="relu",
                          input_shape=params.input_shape,
                      ))
        else:
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=pool_i))

            model.add(Conv2D(
                          filters=n_kerns_i,
                          kernel_size=sizes_i,
                          padding="same",
                          activation="relu",
                     ))
    
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())

    for i,dense_i in enumerate(params.dense_layers):
        model.add(Dense(
                    dense_i,
                    activation="relu",
                    name=f"layer_{i}",
                  ))
        model.add(Dropout(0.5))
    model.add( Dense(params.n_cats, 
                     activation="softmax"))
    if verbose:
        model.summary()

    return ConvNN(model)

@dataclass(frozen=True)
class Hyperparams:
    input_shape:tuple
    n_cats:int
    dense_layers:list 
    n_kerns:list 
    kernel_sizes:list
    pool_size:list 
    
    def __post_init__(self):
        assert len(self.n_kerns) == len(self.kernel_sizes), "n_kerns i kernel_sizes muszą mieć tę samą długość"
        assert len(self.pool_size) == len(self.n_kerns) - 1, "pool_size musi mieć o jeden element mniej niż n_kerns"
    
    def __iter__(self):
        pairs=zip(self.n_kerns,self.kernel_sizes)
        for i,(kerns_i,sizes_i) in enumerate(pairs):
            if(i==0):
                pool_i=None
            else:
                pool_i=self.pool_size[i-1]
            yield kerns_i,sizes_i,pool_i

def minst_params(n_cats=10):
    return Hyperparams(
            input_shape=(28,28,1),
            n_cats=n_cats,
            dense_layers=[1024,512],
            n_kerns=[32,32,64],
            kernel_sizes=[(3,3),(3,3),(3,3)],
            pool_size=[(2,2),(2,2)]
        )

def frame_params(n_cats=20):
    return Hyperparams(
            input_shape=(240, 80, 1),
            n_cats=n_cats,
            dense_layers=[1024,128],
            n_kerns=[32,64,128,256],
            kernel_sizes=[(5, 3),(3,3),(3,3),(3,3)],
            pool_size=[(2, 2),(2, 2),(2, 2)]
        )

def simple_exp(data=None,
               n_neurons=512):
    if(data is None):
        data=base.get_minst_dataset()
    params=minst_params(n_cats=10)
    model=make_cnn(params)
    model.fit(data.train)
    acc=model.eval(data.test)
    print(f"{acc:.4f}")
    return model,data

def cnn_exp( train,
             test,
             params,
             epochs=50):
    model=make_cnn(params)
    model.fit(train,epochs=epochs)
    acc=model.eval(test)
    print(f"{acc:.4f}")
    data=base.DataPair(train,test)
    return model,data   

if __name__ == '__main__':
#    hyper=frame_params()
#    make_cnn(hyper,verbose=True)
    model,data=simple_exp()
#    model.extract(data.train)