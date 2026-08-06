import numpy as np
import deep.cnn
import deep.ae
import argparse
import seq,labels,utils

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
                 dir_path,
                 diff=True):
    nn=get_nn("ae")
    nn_path=f"{dir_path}/ae"
    model_path=f"{nn_path}/model"
    model=nn.read(model_path)
    actions=seq.ActionGroup.read(frame_path)
    def helper(old_frame):
        frame=model.predict(np.expand_dims(old_frame, 0))
        frame=frame.squeeze(axis=(0, 3))
        frame=(frame*255).astype(int)
        if(diff):
            return np.abs(frame-old_frame)
        return new_frame
    actions.lazy_save(helper,f"{nn_path}/reconst")

def extract( frame_path,
             dir_path,
             nn_type="ae",
             layer=0):
    nn=get_nn(nn_type)
    nn_path=f"{dir_path}/{nn_type}"
    model_path=f"{nn_path}/model"
    model=nn.read(model_path)
    seqs=labels.FeatSeqGroup.from_actions( frame_path,
                                           model,
                                           n_layer=layer)
    layer_path=f"{nn_path}/layer_{layer}"
    utils.make_dir(layer_path)
    seqs.save(f"{layer_path}/seqs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str,default="MSR/scaled")
    parser.add_argument("--dir_path", type=str,default="MSR")
    parser.add_argument("--nn_type", type=str,default="ae")
    parser.add_argument("--cmd", type=str,default="extract")
    parser.add_argument("--layer", type=int,default=0)
    args=parser.parse_args()
    if(args.cmd=="train"):
        train( args.frame_path,
	           args.dir_path,
	           args.nn_type)
    if(args.cmd=="reconst"):
        reconstruct( args.frame_path,
                     args.dir_path,)
    if(args.cmd=="extract"):
        extract( args.frame_path,
                 args.dir_path,
                 args.nn_type,
                 args.layer)
