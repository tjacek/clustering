import deep.cnn
import deep.ae

NN_TYPES = { "ae":deep.ae.ConvAE,
             "cnn":deep.cnn.ConvNN}

def make_model(nn_type):
    if(nn_type=="cnn"):
        params=frame_params()
        params["n_cats"]=20
        cnn_factory=deep.cnn.CNNFactory(**params)
        return cnn_factory.build()
    if nn_type == "ae":
        params=frame_params()
        params["latent_dim"]=128
        ae_factory = deep.ae.AEFactory(**params)
        return ae_factory.build()

def frame_params():
    return  { "input_shape":(240, 80, 1),
              "dense_layers":[1024,128],
              "n_kerns":[32,64,128,256],
              "kernel_sizes":[(5, 3),(3,3),(3,3),(3,3)],
              "pool_size":[(2, 2),(2, 2),(2, 2)]}

def minst_params():
    return  { "input_shape":(28,28,1),
              "dense_layers":[1024,512],
              "n_kerns":[32,32,64],
              "kernel_sizes":[(3,3),(3,3),(3,3)],
              "pool_size":[(2,2),(2,2)]}