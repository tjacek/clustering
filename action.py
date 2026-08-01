import numpy as np
import cv2
from collections import defaultdict
from dataclasses import dataclass
import base,cnn,utils

class SeqGroup(list):
    def __init__(self, actions=None):
        if(actions is None):
            actions=[]
        super().__init__(actions)
    
    @classmethod
    def dtype(cls):
        raise NotImplementedError

    def map(self,fun):
        seqs=[seq_i.map(fun) 
                   for seq_i in self]
        return self.__class__(seqs)

    @classmethod
    def read(cls,in_path):
        dtype=cls.dtype()
        seqs=[  dtype.read(path_i) 
                    for path_i in utils.top_files(in_path)]
        return cls(seqs)

    def save(self,out_path):
        utils.make_dir(out_path)
        for seq_i in self:
            seq_i.save(f"{out_path}/{seq_i}")
   
    def flatten(self,fun):
        all_items=[]
        for action_i in self:
            all_items.extend(action_i.eval(fun))
        return all_items

    def split(self,fun=None):
        if(fun is None):
            fun= lambda desc: (desc.person % 2)==1
        train,test=self.__class__([]), self.__class__([])
        for action_i in self:
            if(fun(action_i.desc)):
                train.append(action_i)
            else:
                test.append(action_i)
        return train,test
    
    def as_dataset(self):
        X,y=[],[]
        for action_i in self:
            for frame_j in action_i:
                X.append(frame_j)
                y.append(action_i.desc.cat)
        return base.Dataset(X=np.array(X),
                            y=np.array(y))
    
    @classmethod
    def _from_actions(cls, action_path, model_path, fun):
        actions = ActionGroup.read(action_path)
        model = cnn.ConvNN.read(model_path)
        seqs = cls([])
        for action_i in actions:
            X = np.array(action_i)
            seq_i = cls.dtype()(frames=fun(model, X), desc=action_i.desc)
            seqs.append(seq_i)
        return seqs

    def by_cat(self):
        cat_dict = defaultdict(lambda: self.__class__())
        for seq_i in self:
            cat_i=seq_i.desc.cat
            cat_dict[cat_i].append(seq_i)
        return cat_dict

class Seq(list):
    def __init__( self, 
                  frames,
                  desc):
        super().__init__(frames)
        self.desc=desc
    
    def __str__(self):
        return self.desc.name
    
    def eval(self,fun):
        return [ fun(frame_i) for frame_i in self]
    
    def map(self,fun):
        return self.__class__( frames=self.eval(fun),
                               desc=self.desc )
@dataclass(frozen=True)
class ActionDesc:
    name:str
    cat:int
    person:int  
    
    @classmethod
    def from_path(cls,path):
        name=path.split("/")[-1]
        raw=name.split("_")
        return cls(name=name,
                   cat=int(raw[0])-1, 
                   person=int(raw[1]))

class ActionGroup(SeqGroup):
    @classmethod
    def dtype(cls):
        return Action

    def stats(self,fun):
        values=self.flatten(fun)
        print(f"Mean:{np.mean(values)}")
        print(f"Median:{np.median(values)}")
        print(f"Max:{np.amax(values)}")
        print(f"Min:{np.amin(values)}")

    def rescale( self, 
                 new_width=80,
                 new_height=240):
        def helper(frame):
            return cv2.resize(frame, (new_width, new_height))
        return self.map(helper)

class Action(Seq):

    @classmethod
    def read(cls,in_path):
        frames=[cv2.imread(path_i,cv2.IMREAD_GRAYSCALE)
                    for path_i in utils.top_files(in_path)]
        desc=ActionDesc.from_path(in_path)
        return cls(frames,desc)

    def save(self,out_path):
        utils.make_dir(out_path)
        for i,frame_i in enumerate(self):
            cv2.imwrite(f"{out_path}/{i}.png", frame_i)

def height(frame):
    return frame.shape[0]

def width(frame):
    return frame.shape[1]

if __name__ == '__main__':
    actions=ActionGroup.read("MSR(scaled)")

