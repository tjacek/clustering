import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,BatchNormalization,MaxPooling2D
from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model 

class Experiment(object):
    def __init__(self,dataset,model):
         self.dataset=dataset
         self.model=model
    
    def get_features(self,name_i='dense_1',batch_size=1024):
        ext=self.make_extractor(name_i)
        def helper(x):
            return ext.predict(x,batch_size=batch_size)
        return self.dataset.transform(helper)

    def all_names(self):
        return [layer.name for layer in self.model.layers]

    def make_extractor(self,name_i='dense_1'):
        output= self.model.get_layer(name_i).output 
        return Model(inputs=self.model.input,
                     outputs=output)

class Dataset(object):
    def __init__(self,x_train, y_train,x_test, y_test):
#        if(type(y_train)==list):
#             y_train=tf.one_hot(y_train.astype(np.int32), depth=10)
#        if(type(y_test)==list):
#             y_test=tf.one_hot(y_test.astype(np.int32), depth=10)
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test

    def transform(self,fun):
    	return Dataset(x_train=fun(self.x_train), 
    		           y_train=self.y_train,
    		           x_test=fun(self.x_test), 
    		           y_test=self.y_test)

    def dim(self):
    	return self.x_train.shape

    def get_cat(self,i):
        indices=(self.y_train==i)#[:,i]==1)
        x_i=self.x_train[indices]
        return x_i

class SimpleCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
        self.model.stop_training = True

def get_minst_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return Dataset(x_train, y_train,x_test, y_test)

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
	model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu', data_format='channels_last'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))
	model.add(Dense(1024, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(params['n_cats'], activation='softmax'))
	return model

def default_params():
    return {'n_kern1':32, "kern_size1":(3,3),
            'n_kern2':32, "kern_size2":(3,3),
            'n_kern3':64, "kern_size3":(3,3),  
            "n_cats":10}

def read_exp(in_path,
	         read_dataset=None):
    if(read_dataset is None):
        read_dataset=get_minst_dataset	
    model=load_model(in_path)
    return Experiment(dataset=read_dataset(),
    	              model=model)

def simple_exp(epochs=50,batch_size = 64,out_path=None):
    data=get_minst_dataset()
    params=default_params()
    model=make_cnn(params)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), 
    	          loss='categorical_crossentropy', 
    	          metrics=['acc'])
    callbacks=SimpleCallback()
    history = model.fit(data.x_train, data.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1,
                        callbacks=[callbacks])
    if(not out_path is None):
        model.save(out_path)
    return Experiment(dataset=data,
    	              model=model)


#exp=simple_exp(out_path="simple_cnn")
exp=read_exp("simple_cnn")
feat=exp.get_features('dense')
print(feat.get_cat(1).shape)