import numpy as np
import string
import action,cnn,utils

class SeqGroup(object):
    def __init__(self, seqs):
        self.seqs=seqs
    
    def __len__(self):
        return len(self.seqs)
    
    def __iter__(self):
        return iter(self.seqs)
    
    def save(self,out_path):
        utils.make_dir(out_path)
        for i,seq_i in enumerate(self.seqs):
            seq_i.save(f"{out_path}/{i}")

    @classmethod
    def from_actions( cls,
                      action_path,
                      model_path,
                      n_layer=1):
        actions=action.ActionGroup.read(action_path)
        model=cnn.ConvNN.read(model_path)
        seqs=[]
        for action_i in actions:
            X=np.array(action_i.frames)
            seq_i=Seq( vectors=model.extract(X,n_layer),
                       desc=action_i.desc,
                       labels=model.predict(X))
            print(seq_i.labels)
            print(seq_i)
            print(seq_i.as_symbols())
            seqs.append(seq_i)
        return cls(seqs)

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
        return len(self.vectors)
    
    def __iter__(self):
        return iter(self.vectors)
    
    def __str__(self):
        return self.desc.name

    def save(self,out_path):
        np.savetxt(out_path,self.vectors)

    def as_symbols(self):
        letters = list(string.ascii_lowercase)
        symb_seq=""
        for i in self.labels:
            symb_seq+=letters[i]
        return symb_seq

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

train("MSR/scaled","MSR/model")
seqs=SeqGroup.from_actions("MSR/scaled","MSR/model.keras")
#seqs.save("MSR/seqs")