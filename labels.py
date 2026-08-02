import numpy as np
import string
from collections import defaultdict
import seq,cnn,utils

#class

class FeatSeqGroup(seq.SeqGroup):
    @classmethod
    def dtype(cls):
        return FeatSeq

    @classmethod
    def from_actions(cls, action_path, model_path, n_layer=1):
        return cls._from_actions(action_path, model_path,
                                  lambda model, X: model.extract(X, n_layer))
    def get_pre(self):
        frames=self.flatten(seq.identity)
        indexes=self.flatten(seq.GetIndex())
        order_labeling=self.eval(seq.GetIndex(),
                                 LabelingGroup)
        print(order_labeling[0].as_symbols())

class FeatSeq(seq.Seq):
    @classmethod
    def read(cls,in_path):
        arr=np.load(in_path)
        desc=seq.ActionDesc.from_path(in_path)
        return cls(arr,desc)

    def save(self,out_path):
        np.save(out_path,self)

class LabelingGroup(seq.SeqGroup):
    @classmethod
    def dtype(cls):
        return Labeling

    @classmethod
    def from_actions(cls, action_path, model_path):
        return cls._from_actions(action_path, model_path,
                                  lambda model, X: model.predict(X))
    
    def by_labels(self, seq_group):
        frame_dict=defaultdict(lambda :[])
        label_dict=self.as_dict()
        for seq_i in seq_group:
            labeling_i=label_dict[seq_i.desc.name]
            for label_j,frame_j in zip(labeling_i,seq_i):
                frame_dict[int(label_j)].append(frame_j)
        
        dtype=seq_group.dtype()
        desc=seq.ActionDesc.from_name
        frame_dict={ i:dtype(frames=frames_i,
                             desc=desc(f"{i+1}_0_0"))
                        for i,frames_i in frame_dict.items()}
        return frame_dict

    def unique(self):
        labels=[]
        for seq_i in self:
            labels+=seq_i.unique()
        labels=list(set(labels))
        return np.array(labels)

class Labeling(seq.Seq):
    @classmethod
    def read(cls,in_path):
        arr=np.loadtxt(in_path)
        desc=seq.ActionDesc.from_path(in_path)
        return cls(arr,desc)

    def save(self,out_path):
        np.savetxt(out_path,self)
    
    def unique(self):
        return list(set(self))

    def as_symbols(self):
        letters = list(string.ascii_lowercase)
        n=len(letters)
        symb_seq=""
        def helper(i):
            symb=letters[i%n]
            k= int(np.floor(i/n))
            return symb+str(k)
        for i in self:
            symb_seq+=helper(i)
        return symb_seq

def train( in_path,
           out_path,
           epochs=150):
    actions=seq.ActionGroup.read(in_path)
    train,test=actions.split()
    model,_=cnn.cnn_exp( train.as_dataset(),
                         test.as_dataset(),
                         cnn.frame_params(),
                         epochs=epochs)
    model.save(out_path)

def save_by_labels( label_path,
                    action_path,
                    out_path,
                    mean=True):
    labels=LabelingGroup.read(label_path)
    actions=seq.ActionGroup.read(action_path)
    label_dict=labels.by_labels(actions)
    utils.make_dir(out_path)
    items=label_dict.items()
    for i,(cat_i,group_i) in enumerate(items):
        print(cat_i)
        out_i=f"{out_path}/{i}"
        if(mean):
            group_i.mean_img(out_i)
        else:
            group_i.save(out_i)
#train("MSR/scaled","MSR/model")
#seqs=FeatSeqGroup.from_actions("MSR/scaled","MSR/model.keras")
#seqs.save("MSR/seq")
seqs= FeatSeqGroup.read("MSR/seq")
print(seqs.get_pre())
#labels=LabelingGroup.from_actions("MSR/scaled","MSR/model.keras")
#labels.save("MSR/labels")
#save_by_labels( "MSR/labels",
#                "MSR/scaled",
#                "MSR/mean_img")