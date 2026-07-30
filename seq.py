import numpy as np
import action,cnn

class SeqGroup(object):
    def __init__(self, actions):
        self.seqs=seqs
    
    def __len__(self):
        return len(self.seqs)
    
    def __iter__(self):
        return iter(self.seqs)

class Seq(object):
    def __init__( self,
                  vectors,
                  desc,
                  labels=None):
        if(type(vectors)!=np.ndarray):
            vectors=np.array(vectors)
        if(labels is None):
            labels=[]
        self.vectors = vectors
        self.desc = desc
        self.labels=labels
    
    def __len__(self):
        return len(self.actions)
    
    def __iter__(self):
        return iter(self.actions)
    
    def __str__(self):
        return self.desc.name

def train( in_path,
           out_path,
           epochs=150):
    actions=action.ActionGroup.read(in_path)
    train,test=actions.split()
    model,_=cnn.cnn_exp( train.as_dataset(),
                         test.as_dataset(),
                         cnn.frame_params(),
                         epochs=epochs)
    model.save(out_path)

def make_seqs( action_path,
               model_path):
    actions=action.ActionGroup.read(action_path)
    model=cnn.ConvNN.read(model_path)
    seqs=[]
    for action_i in actions:
        X=np.array(action_i.frames)
        vectors=model.extract(X,n_layer=1)
        labels=model.predict(X)
        seq_i=Seq( vectors=vectors,
                   desc=action_i.desc,
                   labels=labels)
    return SeqGroup(seqs)
    
#train("MSR/scaled","MSR/model")
make_seqs("MSR/scaled","MSR/model.keras")