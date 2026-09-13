import numpy as np
import deep.cnn
import deep.ae
import argparse
import seq#,labels,
import utils


class DirProxy(object):
    def __init__( self,
                  dir_path,
                  n_layer=0):
        self.dir_path=dir_path
        utils.make_dir(self.dir_path)
        self.files={}
        self.n_layer=n_layer

    def __getitem__(self,item):
        if(not item in self.files):
            path=f"{self.dir_path}/{item}"
            utils.make_dir(path)
            self.files[item]=path
        return self.files[item]
    
    @property
    def model(self):
        return self["model"]

    @property
    def layer(self):
        return self[f"layer_{self.n_layer}"]
    
    @property
    def recon(self):
        return self["reconst"]

def train( in_path,
           out_path,
           nn_type="ae",
           epochs=5):
    action_group=seq.get_group("actions")
    actions=action_group.read(in_path)
    train,test=actions.split()
    model=deep.make_model(nn_type)
    model.encoder.summary()
    model.exp( train.as_dataset(),
               test.as_dataset(),
               epochs=epochs)
    nn_dir=DirProxy(f"{out_path}/{nn_type}")
    model.save(nn_dir.model)

def reconstruct( frame_path,
                 dir_path,
                 diff=True):
    nn=deep.NN_TYPES["ae"]
    nn_dir=DirProxy(f"{dir_path}/ae")
    model=nn.read(nn_dir.model)
    action_group=seq.get_group("actions")
    actions=action_group.read(frame_path)
    def helper(old_frame):
        frame=model.predict(np.expand_dims(old_frame, 0))
        frame=frame.squeeze(axis=(0, 3))
        frame=(frame*255).astype(int)
        if(diff):
            return np.abs(frame-old_frame)
        return new_frame
    actions.lazy_save(helper,nn_dir.recon)

def extract( frame_path,
             dir_path,
             nn_type="ae",
             layer=0):
    nn=deep.NN_TYPES[nn_type]
    nn_dir=DirProxy( f"{dir_path}/{nn_type}",
                     layer)
    model=nn.read(nn_dir.model)
    feat_group=seq.get_group("feat")
    seqs=feat_group.from_actions( frame_path,
                                  model,
                                  n_layer=layer)
    seqs.save(f"{nn_dir.layer}/seqs")

def eval( frame_path,
          dir_path,
          nn_type="ae"):
    nn=deep.NN_TYPES[nn_type]
    nn_dir=DirProxy(f"{dir_path}/{nn_type}")
    model=nn.read(nn_dir.model)
    model.model.summary()
    action_group=seq.get_group("actions")
    actions=action_group.read(frame_path)
    train,test=actions.split()
    score=model.eval(test.as_dataset())
    print(f"metric:{score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_path", type=str,default="MSR/scaled")
    parser.add_argument("--dir_path", type=str,default="MSR")
    parser.add_argument("--nn_type", type=str,default="sim")
    parser.add_argument("--cmd", type=str,default="extract")
    parser.add_argument("--layer", type=int,default=1)
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
    if(args.cmd=="eval"):
        eval( args.frame_path,
              args.dir_path,
              args.nn_type)