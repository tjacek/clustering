from tensorflow.keras.layers import Conv2D,Dense,Dropout,Flatten,BatchNormalization,MaxPooling2D
from tensorflow.keras import Input, Model

def make_ae(params):
    input = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, (3, 3), 
                      activation="relu", 
                      padding="same")(input)
    x = MaxPooling2D((2, 2), 
                      padding="same")(x)
    x = Conv2D(32, 
               (3, 3), 
               activation="relu", 
               padding="same")(x)
    x = MaxPooling2D((2, 2), 
                      padding="same")(x)
    x = Conv2DTranspose(32, 
                        (3, 3), 
                        strides=2, 
                        activation="relu", 
                        padding="same")(x)
    x = Conv2DTranspose(32, 
                        (3, 3), 
                        strides=2, 
                        activation="relu", 
                        padding="same")(x)
    x = Conv2D(1, 
               (3, 3), 
               activation="sigmoid", 
               padding="same")(x)
    autoencoder = Model(input, x)
    return autoencoder
#autoencoder.compile(optimizer="adam", loss="binary_crossentropy")