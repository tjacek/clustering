import deep.cnn
import deep.ae
import argparse
import seq,utils

def get_nn(nn_type):
	if(nn_type=="ae"):
		return deep.ae.ConvAE
	if(nn_type=="cnn"):
		return deep.cnn.ConvNN

def train_model( in_path,
                  out_path,
                  nn_type="ae"):
    nn=get_nn(nn_type)
    print(nn)
    actions=seq.ActionGroup.read(in_path)
    train,test=actions.split()
    model,_=nn.make_model( train.as_dataset(),
                           test.as_dataset(),
                           epochs=5)
    model_path=f"{out_path}/{nn_type}"
    utils.make_dir(model_path)
    model.save(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str,default="MSR/scaled")
    parser.add_argument("--dir_path", type=str,default="MSR")
    parser.add_argument("--nn_type", type=str,default="ae")
    parser.add_argument("--cmd", type=str,default="train")
    args=parser.parse_args()
    if(args.cmd=="train"):
        train_model( args.frame_path,
	                 args.dir_path,
	                 args.nn_type)