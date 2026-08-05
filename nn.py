import deep.cnn
import deep.ae
import seq,utils

def get_nn(nn_type):
	if(nn_type=="ae"):
		return deep.ae.ConvAE
	if(nn_type=="cnn"):
		return deep.cnn.ConvNN

def create_model( in_path,
                  out_path,
                  nn_type="ae"):
    nn=get_nn(nn_type)
    print(nn)
    actions=seq.ActionGroup.read(in_path)
    train,test=actions.split()
    model,_=nn.train_model( train.as_dataset(),
                    test.as_dataset(),
                    deep.ae.frame_ae_params(),
                    epochs=5)
    model_path=f"{out_path}/{nn_type}"
    utils.make_dir(model_path)
    model.save(model_path)

create_model( "MSR/scaled",
	          "MSR",
	          nn_type="ae")