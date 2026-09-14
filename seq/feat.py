import numpy as np
from dataclasses import dataclass
from collections import namedtuple
import seq.core as core
import base

class FeatSeqGroup(core.SeqGroup):
    @classmethod
    def dtype(cls):
        return FeatSeq

    @classmethod
    def from_actions( cls, 
                      action_path, 
                      model, 
                      n_layer=1):
        def helper(action):
            X=np.array(action)
            return model.extract(X, n_layer)
        return core.lazy_convert( action_path,
                                  helper,
                                  new_type=FeatSeqGroup,
                                  old_type=core.Action)

    def as_precluster(self):
        return Preclustering.from_feats(self)

    def dim(self):
        return self[0][0].shape

    def group_info( self, 
                   label_group):
        frame_dict,cluster_info=self.group(label_group)
        frame_dict=base.SmartDict(frame_dict)  
        def helper(i,frames_i):
            raw_i=list(zip(*cluster_info[i]))
            names=["order","cat","person","names"]
            info_i=dict(zip(names,raw_i))
            info_i["cat"]=np.array(info_i["cat"],dtype=int)
            return FrameInfo( np.array(frames_i),
                              info_i)
        return frame_dict.map(helper)

class FeatSeq(core.Seq):
    @classmethod
    def read(cls,in_path):
        arr=np.load(in_path)
        desc=core.ActionDesc.from_path(in_path)
        return cls(arr,desc)

    def save(self,out_path):
        np.save(out_path,self)
    
    def as_numpy(self):
        return np.array(self,dtype=float)

    def distance(self):
        n=len(self)-1
        return [ np.linalg.norm(self[i+1]-self[i],ord=2) 
                  for i in range(n)]

class FrameInfo:
    def __init__( self,
                  frames,
                  info_dict):
        self.frames=frames
        self.info_dict=info_dict

    def __getitem__(self,item):
        return self.info_dict[item]#getattr(self,item)
    
    def __setitem__(self, item, value):
        self.info_dict[item]=value

    def unique(self,item):
        return list(set(self[item]))
    
    def info(self,item):
        data=self[item]
        unique=self.unique(item)
        index={ type_i:i for i,type_i in enumerate(unique)}
        Info=namedtuple("Info",["data", "unique","index"])
        return Info(data,unique,index)

    def discretize(self,n=10):
        self["order"]=n*np.array(self["order"])
        self["order"]=np.floor(self["order"])
        self["order"]=self["order"].astype(int)

