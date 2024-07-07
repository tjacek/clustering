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
	model.add(Dense(params['n_cata'], activation='softmax'))
	return model

def default_params():
    return {'n_kern1':32, "kern_size1":(3,3),
            'n_kern2':32, "kern_size2":(3,3),
            'n_kern3':64, "kern_size3":(3,3),  
            "n_cats":10}
