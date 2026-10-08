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
from sklearn.metrics import accuracy_score,classification_report
from tensorflow.keras.models import load_model 
import keras
import deep.core as core
from dataclasses import dataclass
import itertools
import base

@keras.saving.register_keras_serializable(package="deep.sim")
class EuclideanDistance(Layer):
    def call(self, inputs):
        emb_a, emb_b = inputs
        sum_sq = tf.reduce_sum(tf.square(emb_a - emb_b), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_sq, 1e-9))

@keras.saving.register_keras_serializable(package="deep.sim")
class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(self.margin - y_pred, 0.0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin})
        return config

def _make_pairs(X, y):
    raise Exception(X[0].shape)
    rng = np.random.default_rng()
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
 
    pairs_a, pairs_b, labels = [], [], []
 
    for i in range(len(X)):
        cls_i = y[i]
 
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
    PROTO="prototypes.npz"
    def __init__(  self, 
                   model, 
                   meta, 
                   encoder,
                   prototypes=None):
        super().__init__(model, meta)
        self.encoder = encoder
        self.prototypes = prototypes

    @classmethod
    def read(cls,in_path):
        nn_meta=core.NNMeta.read(f"{in_path}/{cls.META_FILE}")
        model = load_model(f"{in_path}/{cls.MODEL_FILE}")
        encoder=model.get_layer("shared_encoder")
        
        data = np.load(f"{in_path}/{cls.PROTO}", allow_pickle=True)
        prototypes = dict(zip(data["classes"], data["embeddings"]))
        model.summary()
        return SiameseNN( model,
                          nn_meta,
                          encoder,
                          prototypes)
    def save(self,out_path):
        super(SiameseNN, self).save(out_path)
        embd=list(self.prototypes.values())
        np.savez(f"{out_path}/{self.PROTO}",
                 classes=list(self.prototypes.keys()),
                 embeddings=np.stack(embd))

    def fit( self, 
             data_pairs, 
             epochs=50, 
             batch_size=64, 
             margin=1.0):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-6),
            loss=ContrastiveLoss(margin=margin),
        )
 
        callbacks = [DistCallback()]

        self.model.fit(
            data_pairs.pairs,
            data_pairs.labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
        )
        self.nn_meta.n_epochs += epochs
 
    def make_prototypes(self,data):
        X,y=data.X,data.y
        emb = self.encoder.predict(X, batch_size=256, verbose=0)
        self.prototypes = {
            cls_i: emb[y == cls_i].mean(axis=0) for cls_i in np.unique(y)
        }
 
    def predict(self, X):
        emb = self.encoder.predict(X, batch_size=256, verbose=0)
        classes = list(self.prototypes.keys())
        prot= np.array([self.prototypes[c] for c in classes])
        y_pred=[]
        for emb_i in emb:
            dist_i = np.linalg.norm(prot - emb_i, axis=1)
            y_pred.append(np.argmin(dist_i))
        return y_pred

    def eval(self, data):
        y_pred = self.predict(data.X)
        print(classification_report(data.y,y_pred))
        return accuracy_score(data.y, y_pred)
 
    def extract(self, data, n_layer=1):
        old_X = data.X if isinstance(data, base.Dataset) else data
        X = np.expand_dims(old_X.astype("float32") / 255.0, -1)
        if(self.extractor is None or 
              self.extractor_layer!=n_layer):
            layer = self.encoder.get_layer(f"layer_{n_layer}")
            self.extractor = Model(
                                inputs=self.encoder.inputs,
                                outputs=layer.output,
                              )
            self.n_layer=n_layer
        feat = self.extractor.predict(X, batch_size=64, verbose=0)
        return feat
 
    def init_extractor(self, n_layer):
        layer_output = self.encoder.layers[n_layer].output
        return Model(inputs=self.encoder.inputs, outputs=layer_output)
 
    def exp(self, train, test, epochs=50):
        data_pairs=make_pairs(train)
        data_pairs.rescale()
        self.fit(data_pairs, epochs=epochs)
        data_train=train.as_dataset()
        data_train.rescale()
        self.make_prototypes(data_train)
        data_test=test.as_dataset()
        data_test.rescale()
        acc = self.eval(data_test)
        print(f"{acc:.4f}")

def make_pairs(actions):
    n_actions=range(len(actions))
    pairs=itertools.combinations(n_actions, r=2)
    x,y,labels=[],[],[]
    for i,j in pairs:
        if( (i%2)==0 or (j%2)==0 ):
            continue
        action_i=actions[i]
        action_j=actions[j]
        x.append(action_i.midpoint())
        y.append(action_j.midpoint())
        cat_i=action_i.desc.cat
        cat_j=action_j.desc.cat
        labels.append(int(cat_i==cat_j))
    pairs=(np.array(x),np.array(y))
    return PairsData(pairs,np.array(labels))

@dataclass
class PairsData:
    pairs:tuple
    labels:np.ndarray

    def rescale(self):
        x= self.pairs[0].astype("float32")/255.0
        y= self.pairs[1].astype("float32")/255.0
        self.pairs=(x,y)

class SiameseFactory(core.NNFactory):
 
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
 
#        encoder.add(Dense(self.embedding_dim, name="embedding"))
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