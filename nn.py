import numpy as np
import deep.cnn
import deep.ae
import argparse
import seq,utils

def get_nn(nn_type):
	if(nn_type=="ae"):
		return deep.ae.ConvAE
	if(nn_type=="cnn"):
		return deep.cnn.ConvNN

def train( in_path,
           out_path,
           nn_type="ae"):
    nn=get_nn(nn_type)
    actions=seq.ActionGroup.read(in_path)
    train,test=actions.split()
    model,_=nn.make_model( train.as_dataset(),
                           test.as_dataset(),
                           epochs=5)
    model_path=f"{out_path}/{nn_type}"
    utils.make_dir(model_path)
    model.save(model_path)

def reconstruct( frame_path,
                 dir_path):
    nn=get_nn("ae")
    nn_path=f"{dir_path}/ae"
    model_path=f"{nn_path}/model"
    model=nn.read(model_path)
    actions=seq.ActionGroup.read(frame_path)
    def helper(frame):
        frame=np.expand_dims(frame, 0)
        new_frame=model.predict(frame)
        return frame
    reconst_actions=actions.map(helper)
    print(len(actions))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str,default="MSR/scaled")
    parser.add_argument("--dir_path", type=str,default="MSR")
    parser.add_argument("--nn_type", type=str,default="ae")
    parser.add_argument("--cmd", type=str,default="reconst")
    args=parser.parse_args()
    if(args.cmd=="train"):
        train( args.frame_path,
	           args.dir_path,
	           args.nn_type)
    if(args.cmd=="reconst"):
        reconstruct( args.frame_path,
                     args.dir_path,)