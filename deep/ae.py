import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    BatchNormalization,
    MaxPooling2D,
    UpSampling2D,
    Reshape,
    Flatten,
    GlobalAveragePooling2D,
)
from dataclasses import dataclass
import re
import deep.core
import base,utils

class ConvAE(deep.core.NeuralModel): 
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder
        self._extractor = None
        self.extractor_layer = None

    @classmethod
    def build(cls, params=None, verbose=False):
        if(params is None):
            params=frame_ae_params()
        return build_autoencoder(params, verbose)

    def fit(self, data, epochs=50, batch_size=64):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="mse",
        )
 
        callbacks = [SimpleAECallback()]
 
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
   
    def eval(self, data):
        X = np.expand_dims(data.X.astype("float32") / 255.0, -1)
        X_pred = self.model.predict(X, verbose=0)
        return float(np.mean((X - X_pred) ** 2))
 
    def encode(self, data):
        old_X = data.X if isinstance(data, base.Dataset) else data
        X = np.expand_dims(old_X.astype("float32") / 255.0, -1)
        return self.encoder.predict(X, batch_size=256, verbose=0)

    def extract(self, data, n_layer=1):
        old_X = data.X if isinstance(data, base.Dataset) else data
        X = np.expand_dims(old_X.astype("float32") / 255.0, -1)
 
        if ( self._extractor is None or 
             self.extractor_layer != n_layer):
            layer_names=self.find_layers(r"^layer_\d+")
            if(len(layer_names)<=n_layer):
                name="bottleneck"
            else:
                name=layer_names[n_layer]
            layer = self.model.get_layer(name)
            self._extractor = Model(inputs=self.model.inputs, outputs=layer.output)
            self.extractor_layer = n_layer
 
        return self._extractor.predict(X, batch_size=256, verbose=0)

    def predict(self, X):
        X = np.expand_dims(X.astype("float32") / 255.0, -1)
        return self.model.predict(X, verbose=0)
 
    def save(self, out_path):
        utils.make_dir(out_path)
        self.model.save(f"{out_path}/full.keras")
        self.encoder.save(f"{out_path}/encoder.keras")
 
    @classmethod
    def read(cls, in_path):
        model = tf.keras.models.load_model(f"{in_path}/full.keras")
        encoder = tf.keras.models.load_model(f"{in_path}/encoder.keras")
        return cls(model, encoder)

    @classmethod
    def make_model( cls, 
              train,
              test,
              params=None,
              epochs=50):
        model = cls.build(params)
        model.fit(train, epochs=epochs)
        mse = model.eval(test)
        print(f"{mse:.4f}")
        data = base.DataPair(train, test)
        return model, data

class SimpleAECallback(tf.keras.callbacks.Callback):
 
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

def build_autoencoder(params=None, verbose=False):
    if params is None:
        params = minst_ae_params()
 
    inputs = Input(shape=params.input_shape)
    x = inputs
 
    # --- Enkoder ---
    shapes_before_pool = []  # potrzebne, by symetrycznie odtworzyć rozmiary w dekoderze
    for i, (n_kerns_i, sizes_i, pool_i) in enumerate(params):
        x = Conv2D(
            filters=n_kerns_i,
            kernel_size=sizes_i,
            padding="same",
            activation="relu",
            name=f"enc_conv_{i}",
        )(x)
        x = BatchNormalization()(x)
        shapes_before_pool.append(x.shape[1:3])
        if pool_i is not None:
            x = MaxPooling2D(pool_size=pool_i, padding="same")(x)
 
    pre_bottleneck_shape = x.shape[1:]  # (H, W, C) tuż przed spłaszczeniem
    x = Flatten()(x)
 
    for i, dense_i in enumerate(params.dense_layers):
        x = Dense(dense_i, activation="relu", name=f"layer_{i}")(x)
        x = Dropout(0.3)(x)
 
    bottleneck = Dense(params.latent_dim, activation="relu", name="bottleneck")(x)
 
    encoder = Model(inputs=inputs, outputs=bottleneck, name="encoder")
 
    # --- Dekoder (odbicie lustrzane enkodera) ---
    x = bottleneck
    for i, dense_i in enumerate(reversed(params.dense_layers)):
        x = Dense(dense_i, activation="relu", name=f"dec_layer_{i}")(x)
        x = Dropout(0.3)(x)
 
    x = Dense(int(np.prod(pre_bottleneck_shape)), activation="relu")(x)
    x = Reshape(pre_bottleneck_shape)(x)
 
    rev_kerns = list(reversed(params.n_kerns))
    rev_sizes = list(reversed(params.kernel_sizes))
    rev_pools = list(reversed(params.pool_size)) + [None]  # brak poolingu przy wejściu
 
    for i, (n_kerns_i, sizes_i, pool_i) in enumerate(zip(rev_kerns, rev_sizes, rev_pools)):
        if pool_i is not None:
            x = UpSampling2D(size=pool_i)(x)
        x = Conv2DTranspose(
            filters=n_kerns_i,
            kernel_size=sizes_i,
            padding="same",
            activation="relu",
            name=f"dec_conv_{i}",
        )(x)
        x = BatchNormalization()(x)
 
    n_channels = params.input_shape[-1]
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
 
    return ConvAE(autoencoder, encoder)

@dataclass(frozen=True)
class AEHyperparams:
    input_shape: tuple
    latent_dim: int
    dense_layers: list
    n_kerns: list
    kernel_sizes: list
    pool_size: list
 
    def __post_init__(self):
        assert len(self.n_kerns) == len(self.kernel_sizes), \
            "n_kerns i kernel_sizes muszą mieć tę samą długość"
        assert len(self.pool_size) == len(self.n_kerns) - 1, \
            "pool_size musi mieć o jeden element mniej niż n_kerns"
 
    def __iter__(self):
        pairs = zip(self.n_kerns, self.kernel_sizes)
        for i, (kerns_i, sizes_i) in enumerate(pairs):
            if i == 0:
                pool_i = None
            else:
                pool_i = self.pool_size[i - 1]
            yield kerns_i, sizes_i, pool_i
 
 
def minst_ae_params(latent_dim=64):
    return AEHyperparams(
        input_shape=(28, 28, 1),
        latent_dim=latent_dim,
        dense_layers=[512],
        n_kerns=[32, 32, 64],
        kernel_sizes=[(3, 3), (3, 3), (3, 3)],
        pool_size=[(2, 2), (2, 2)],
    )
 
 
def frame_ae_params(latent_dim=128):
    return AEHyperparams(
        input_shape=(240, 80, 1),
        latent_dim=latent_dim,
        dense_layers=[256],
        n_kerns=[32, 64, 128, 256],
        kernel_sizes=[(5, 3), (3, 3), (3, 3), (3, 3)],
        pool_size=[(2, 2), (2, 2), (2, 2)],
    )
