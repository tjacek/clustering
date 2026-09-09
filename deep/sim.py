import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import (
    Layer,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from sklearn.metrics import accuracy_score
 
import deep.core as core
from deep import base

class EuclideanDistance(Layer):
    def call(self, inputs):
        emb_a, emb_b = inputs
        sum_sq = tf.reduce_sum(tf.square(emb_a - emb_b), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_sq, 1e-9))

def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0.0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss


def make_pairs(X, y):
    rng = np.random.default_rng()
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
 
    pairs_a, pairs_b, labels = [], [], []
 
    for i in range(len(X)):
        cls = y[i]
 
        j = rng.choice(idx_by_class[cls])
        pairs_a.append(X[i])
        pairs_b.append(X[j])
        labels.append(1)
 
        neg_cls = rng.choice(classes[classes != cls])
        k = rng.choice(idx_by_class[neg_cls])
        pairs_a.append(X[i])
        pairs_b.append(X[k])
        labels.append(0)
 
    return (
        np.array(pairs_a, dtype="float32"),
        np.array(pairs_b, dtype="float32"),
        np.array(labels, dtype="float32"),
    )
 
 class DistCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_thres=0.02):
        self.loss_thres = loss_thres
 
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
 
        if loss is not None and loss < self.loss_thres:
            print(f"\nOsiągnięto stratę < {self.loss_thres}, zatrzymuję trening.")
            self.model.stop_training = True

class SiameseNN(core.NeuralModel):
    def __init__(self, model, meta, encoder):
        super().__init__(model, meta)
        self.encoder = encoder
        self.prototypes = None
 
    def fit(self, data, epochs=50, batch_size=64, margin=1.0):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=contrastive_loss(margin=margin),
        )
 
        callbacks = [DistCallback()]
 
        X = np.expand_dims(data.X.astype("float32") / 255.0, -1)
        X_a, X_b, pair_y = make_pairs(X, data.y)
 
        self.model.fit(
            [X_a, X_b],
            pair_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
        )
        self.nn_meta.n_epochs += epochs
 
        self._fit_prototypes(X, data.y)
 
    def _fit_prototypes(self, X, y):
        emb = self.encoder.predict(X, batch_size=256, verbose=0)
        self.prototypes = {
            cls: emb[y == cls].mean(axis=0) for cls in np.unique(y)
        }
 
    def predict(self, X):
        X = X.astype("float32") / 255.0
        if X.ndim == 3:
            X = np.expand_dims(X, -1)
        emb = self.encoder.predict(X, batch_size=256, verbose=0)
 
        classes = np.array(list(self.prototypes.keys()))
        protos = np.stack(list(self.prototypes.values()))
 
        dists = np.linalg.norm(emb[:, None, :] - protos[None, :, :], axis=2)
        return classes[np.argmin(dists, axis=1)]
 
    def eval(self, data):
        y_pred = self.predict(data.X)
        return accuracy_score(data.y, y_pred)
 
    def extract(self, data, n_layer=1):
        old_X = data.X if isinstance(data, base.Dataset) else data
        X = np.expand_dims(old_X.astype("float32") / 255.0, -1)
        extr = self.init_extractor(n_layer)
        feat = extr.predict(X, batch_size=256, verbose=0)
        return feat
 
    def init_extractor(self, n_layer):
        layer_output = self.encoder.layers[n_layer].output
        return Model(inputs=self.encoder.inputs, outputs=layer_output)
 
    def exp(self, train, test, epochs=50):
        self.fit(train, epochs=epochs)
        acc = self.eval(test)
        print(f"{acc:.4f}")

class SiameseFactory(core.NNFactory):
    embedding_dim: int = 20
 
    def build_encoder(self):
        encoder = Sequential(name="shared_encoder")
        encoder.add(self.input_layer())
        encoder.add(self.get_conv(0))
 
        for i in range(self.n_conv - 1):
            encoder.add(BatchNormalization())
            encoder.add(self.get_pool(i))
            encoder.add(self.get_conv(i + 1))
        encoder.add(BatchNormalization())
        encoder.add(GlobalAveragePooling2D())
 
        for i in range(self.n_dense):
            encoder.add(self.get_dense(i))
            encoder.add(Dropout(0.5))
 
        encoder.add(Dense(self.embedding_dim, name="embedding"))
        return encoder
 
    def build(self, verbose=False):
        encoder = self.build_encoder()
        input_shape = encoder.input_shape[1:]
 
        input_a = Input(shape=input_shape, name="input_a")
        input_b = Input(shape=input_shape, name="input_b")
 
        emb_a = encoder(input_a)
        emb_b = encoder(input_b)
        distance = EuclideanDistance(name="euclidean_distance")([emb_a, emb_b])
 
        model = Model([input_a, input_b], distance, name="siamese_network")
 
        if verbose:
            encoder.summary()
            model.summary()
 
        meta = core.NNMeta("SiameseNN", "SiameseFactory", self.__dict__)
        return SiameseNN(model, meta, encoder)