import numpy as np
from dataclasses import dataclass
import seq.core
import base

class FeatSeqGroup(seq.core.SeqGroup):
    @classmethod
    def dtype(cls):
        return FeatSeq

    @classmethod
    def from_actions(cls, action_path, model, n_layer=1):
        return cls._from_actions(action_path, model,
                                  lambda  X: model.extract(X, n_layer))

    def as_precluster(self):
        return Preclustering.from_feats(self)

    def dim(self):
        return self[0][0].shape

    def group_info( self, 
                   label_group):
        frame_dict,info_dict=self.group(label_group)
        frame_dict=base.SmartDict(frame_dict)  
        def helper(i,frames_i):
            info_i=info_dict[i]
            order,cat,person=list(zip(*info_i))
            cat=np.array(cat,dtype=int)
            frames_i=np.array(frames_i)
            return FrameInfo(frames_i,cat,order,person)
        return frame_dict.map(helper)

class FeatSeq(seq.core.Seq):
    @classmethod
    def read(cls,in_path):
        arr=np.load(in_path)
        desc=seq.core.ActionDesc.from_path(in_path)
        return cls(arr,desc)

    def save(self,out_path):
        np.save(out_path,self)
    
    def as_numpy(self):
        return np.array(self,dtype=float)

    def distance(self):
        n=len(self)-1
        return [ np.linalg.norm(self[i+1]-self[i],ord=2) 
                  for i in range(n)]

@dataclass
class FrameInfo:
    frames:np.ndarray
    cat: list
    order:list
    person: list

    def __getitem__(self,item):
        return getattr(self,item)

    def discretize(self,n=10):
        self.order=n*np.array(self.order)
        self.order=np.floor(self.order)
        self.order=self.order.astype(int)