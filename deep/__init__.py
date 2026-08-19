import deep.cnn
import deep.ae

NN_TYPES = { #"ae":deep.cnn.ConvAE,
             "cnn":deep.cnn.ConvNN}


def make_model(nn_type):
	if(nn_type=="cnn"):
		cnn_factory=deep.cnn.CNNFactory()
		return cnn_factory.build()
