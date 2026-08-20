import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from dataclasses import dataclass
from tensorflow.keras.layers import (
#    Conv2D,
    Dense,
    Dropout,
    BatchNormalization,
    Flatten,
    Conv2DTranspose,
    UpSampling2D,
#    MaxPooling2D,
    GlobalAveragePooling2D,
    Reshape,
)
import deep.core

class ConvAE(deep.core.NeuralModel): 
    def __init__(  self, 
    	           model, 
    	           encoder,
    	           meta):
        super().__init__(model,meta)
        self.encoder = encoder

    def fit( self, 
    	     data, 
    	     epochs=50, 
    	     batch_size=64):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="mse",
        )
 
        callbacks = [MSECallback()]
 
        X = np.expand_dims(data.X.astype("float32") / 255.0, -1)
 
        self.model.fit(
            X,
            X,                       
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
        )
        self.nn_meta.n_epochs+=epochs

    def eval(self, data):
        X = np.expand_dims(data.X.astype("float32") / 255.0, -1)
        X_pred = self.model.predict(X, verbose=0)
        return float(np.mean((X - X_pred) ** 2))        

    def exp( self, 
             train,
             test,
             epochs=50):
        self.fit(train,epochs=epochs)
        mse=self.eval(test)
        print(f"MSE{mse:.4f}")

class MSECallback(tf.keras.callbacks.Callback):
 
    def __init__(self, patience=10, min_delta=1e-5):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.wait = 0
 
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("val_loss", logs.get("loss"))
        if loss is None:
            return
 
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nBrak poprawy przez {self.patience} epok, zatrzymuję trening.")
                self.model.stop_training = True

@dataclass
class AEFactory(deep.core.NNFactory):
    latent_dim: int
    
    def get_bottleneck(self):
        return Dense(self.latent_dim, activation="relu", name="bottleneck")
    
    def get_dec_dense(self, i):
        j = self.n_dense - 1 - i
        return deep.core.dense_layer(self.dense_layers[j], f"dec_layer_{i}")
    
    def rev_kerns(self):
        return list(reversed(self.n_kerns))

    def rev_sizes(self):
        return list(reversed(self.kernel_sizes))

    def rev_pool_indices(self):
        return list(reversed(range(len(self.pool_size)))) + [None]    
    def get_upsample(self, i):
        return UpSampling2D(size=self.pool_size[i])
    
    def build(self, verbose=False):
        inputs = self.input_layer()
        x = self.get_conv(0)(inputs)
        x = BatchNormalization()(x)
        for i in range(self.n_conv - 1):
            x = self.get_pool(i)(x)
            x = self.get_conv(i + 1)(x)
            x = BatchNormalization()(x)

        pre_bottleneck_shape = x.shape[1:]
        x = Flatten()(x)

        for i in range(self.n_dense):
            x = self.get_dense(i)(x)
            x = Dropout(0.3)(x)

        bottleneck = self.get_bottleneck()(x)
        encoder = Model(inputs=inputs, outputs=bottleneck, name="encoder")

        # --- Dekoder ---
        x = bottleneck
        for i in range(self.n_dense):
            x = self.get_dec_dense(i)(x)
            x = Dropout(0.3)(x)

        x = Dense(int(np.prod(pre_bottleneck_shape)), activation="relu")(x)
        x = Reshape(pre_bottleneck_shape)(x)

        rev_kerns = self.rev_kerns()
        rev_sizes = self.rev_sizes()
        rev_pool_idx = self.rev_pool_indices()

        for i, (filters_i, size_i, pool_idx_i) in enumerate(
                zip(rev_kerns, rev_sizes, rev_pool_idx)):
            if pool_idx_i is not None:
                x = self.get_upsample(pool_idx_i)(x)
            x = deep.core.deconv_layer(i, filters_i, size_i)(x)
            x = BatchNormalization()(x)

        n_channels = self.input_shape[-1]
        outputs = Conv2DTranspose(
                                  filters=n_channels,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  activation="sigmoid",   # wyjście w [0,1], zgodnie z normalizacją /255.0
                                  name="reconstruction",
                  )(x)

        autoencoder = Model(inputs=inputs, outputs=outputs, name="autoencoder")

        if verbose:
            encoder.summary()
            autoencoder.summary()

        meta = deep.core.NNMeta("ConvAE", "AEFactory", self.__dict__)
        return ConvAE(autoencoder, encoder, meta)



class _AEFactory(object):
    def __init__(self, hyper=None):
        if hyper is None:
            hyper = deep.core.frame_params()
        self.hyper = hyper

    def build(self, verbose=True):
        hyper = self.hyper

        inputs = hyper.input_layer()
        x = inputs
        x = hyper.conv_layer(0)(x)
        for i in range(hyper.n_conv-1):
            x = hyper.conv_layer(i+1)(x)
            x = BatchNormalization()(x)
#            if i < hyper.n_conv - 1:
            x = hyper.pool_layer(i)(x)

        pre_bottleneck_shape = x.shape[1:]
        x = Flatten()(x)

        for i in range(hyper.n_dense):
            x = hyper.dense_layer(i)(x)
            x = Dropout(0.3)(x)

        bottleneck = bottleneck_layer(64)(x)
        encoder = Model(inputs=inputs, outputs=bottleneck, name="encoder")

        # --- Dekoder (odbicie lustrzane enkodera) ---
        x = bottleneck
        for i in range(hyper.n_dense):
            j = hyper.n_dense - 1 - i
            x = Dense(hyper.dense_layers[j], activation="relu", name=f"dec_layer_{i}")(x)
            x = Dropout(0.3)(x)

        x = Dense(int(np.prod(pre_bottleneck_shape)), activation="relu")(x)
        x = Reshape(pre_bottleneck_shape)(x)

        rev_kerns = hyper.rev_kerns()
        rev_sizes = hyper.rev_sizes()
        rev_pool_idx = hyper.rev_pool_indices()

        for i, (filters_i, size_i, pool_idx_i) in enumerate(
                zip(rev_kerns, rev_sizes, rev_pool_idx)):
            if pool_idx_i is not None:
                x = hyper.upsample_layer(pool_idx_i)(x)
            x = deconv_layer(i,filters_i, size_i)(x)#hyper.deconv_layer(i, filters_i, size_i)(x)
            x = BatchNormalization()(x)

        n_channels = hyper.input_shape[-1]
        outputs = Conv2DTranspose(
                    filters=n_channels,
                    kernel_size=(3, 3),
                    padding="same",
                    activation="sigmoid",   # wyjście w [0,1], zgodnie z normalizacją /255.0
                    name="reconstruction",
        )(x)
 
#        outputs = hyper.output_layer()(x)

        autoencoder = Model(inputs=inputs, outputs=outputs, name="autoencoder")

        if verbose:
            encoder.summary()
            autoencoder.summary()

        meta = deep.core.NNMeta("ConvAE", "AEFactory", hyper.__dict__)
        return ConvAE(autoencoder, encoder, meta)


def bottleneck_layer(latent_dim):
	return Dense(latent_dim, activation="relu", name="bottleneck")

