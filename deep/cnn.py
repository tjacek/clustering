import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
#    Conv2D,
    Dense,
    Dropout,
    BatchNormalization,
#    MaxPooling2D,
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
        self.nn_meta.n_epochs+=epochs

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
        feat = extr.predict(X, batch_size=256, verbose=0)
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

class CNNFactory(deep.core.NNFactory):
    n_cats: int = 0
    def build(self, verbose=False):
        model = Sequential()
        model.add(self.input_layer())
        model.add(self.get_conv(0))

        for i in range(self.n_conv-1):
            model.add(BatchNormalization())
            model.add(self.get_pool(i))
            model.add(self.get_conv(i+1))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())

        for i in range(self.n_dense):
            model.add(self.get_dense(i))
            model.add(Dropout(0.5))

        model.add( Dense(self.n_cats, 
                         activation="softmax"))
        if verbose:
           model.summary()
        meta=deep.core.NNMeta( "ConvNN",
                           "ConvBuilder",
                           self.__dict__)
        return ConvNN(model,meta)