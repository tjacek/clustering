import deep.cnn
import deep.ae

NN_TYPES = { "ae":deep.ae.ConvAE,
             "cnn":deep.cnn.ConvNN}

def make_model(nn_type):
    if(nn_type=="cnn"):
        cnn_factory=deep.cnn.CNNFactory( input_shape=(240, 80, 1),
                                         n_cats=20,
                                         dense_layers=[1024,128],
                                         n_kerns=[32,64,128,256],
                                         kernel_sizes=[(5, 3),(3,3),(3,3),(3,3)],
                                         pool_size=[(2, 2),(2, 2),(2, 2)])
        return cnn_factory.build()
    if nn_type == "ae":
        ae_factory = deep.ae.AEFactory( input_shape=(240, 80, 1),
                                        latent_dim=128,
                                        n_cats=20,
                                        dense_layers=[256],
                                        n_kerns=[32, 64, 128, 256],
                                        kernel_sizes=[(5, 3), (3, 3), (3, 3), (3, 3)],
                                        pool_size=[(2, 2), (2, 2), (2, 2)],)
        return ae_factory.build()