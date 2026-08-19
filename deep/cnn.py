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
import deep.core
import base

class ConvNN(deep.core.NeuralModel):
    def fit( self, 
             data, 
             epochs=50, 
             batch_size=64):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        callbacks = [AccCallback()]

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
    
    def predict(self,X):
        X=X.astype("float32") / 255.0
        prob= self.model.predict(X,verbose=0)
        return np.argmax(prob,axis=1)
    
    def eval(self, data):
        X = np.expand_dims(data.X.astype("float32") / 255.0, -1)

        y_pred = self.model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred, axis=1)

        return accuracy_score(data.y, y_pred)
    
    def extract(self, data, n_layer=1):
        old_X = data.X if(isinstance(data, base.Dataset)) else data
        X = np.expand_dims(old_X.astype("float32") / 255.0, -1)
        extr=self.init_extractor(n_layer)
        feat = self.extr.predict(X, batch_size=256, verbose=0)
        return feat
    
    def exp( self, 
             train,
             test,
             epochs=50):
        self.fit(train,epochs=epochs)
        acc=self.eval(test)
        print(f"{acc:.4f}")

class AccCallback(tf.keras.callbacks.Callback):
    def __init__(self, acc_thres=0.995):
        self.acc_thres=acc_thres

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy")

        if acc is not None and acc > self.acc_thres:
            print("\nReached 99.5% accuracy, stopping training.")
            self.model.stop_training = True

@dataclass(frozen=True)
class Hyperparams:
    input_shape:tuple
    n_cats:int
    dense_layers:list 
    n_kerns:list 
    kernel_sizes:list
    pool_size:list 
    
    def __post_init__(self):
        assert len(self.n_kerns) == len(self.kernel_sizes), "n_kerns and kernel_sizes must have the same length"
        assert len(self.pool_size) == len(self.n_kerns) - 1, "pool_size musi mieć o jeden element mniej niż n_kerns"
    
    @property
    def n_conv(self):
        return len(self.n_kerns)
    
    @property
    def n_dense(self):
        return len(self.dense_layers)   
    
    def conv_layer(self,i):
        args={ "filters":self.n_kerns[i],
               "kernel_size":self.kernel_sizes[i],
               "padding":"same",
               "activation":"relu"}
        if(i==0):
            args["input_shape"]=self.input_shape
        return Conv2D(**args)

    def dense_layer(self,i):
        return Dense( self.dense_layers[i],
                      activation="relu",
                      name=f"layer_{i}")

    def pool_layer(self,i):
        return MaxPooling2D(pool_size=self.pool_size[i])


class CNNFactory(object):
    def __init__(self,hyper=None):
        if(hyper is None):
            hyper=frame_params()
        self.hyper=hyper

    def build(self, verbose=False):
        model = Sequential()
        for i in range(self.hyper.n_conv):
            if(i!=0):
                model.add(BatchNormalization())
                model.add(self.hyper.pool_layer(i-1))
            model.add(self.hyper.conv_layer(i))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        
        for i in range(self.hyper.n_dense):
            model.add(self.hyper.dense_layer(i))
            model.add(Dropout(0.5))

        model.add( Dense(self.hyper.n_cats, 
                         activation="softmax"))
        if verbose:
           model.summary()
        meta=deep.core.NNMeta( "ConvNN",
                           "ConvBuilder",
                           self.hyper.__dict__)
        return ConvNN(model,meta)

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