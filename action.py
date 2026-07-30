import numpy as np
import cv2
from dataclasses import dataclass
import base,utils

class ActionGroup(object):
    def __init__(self, actions):
        self.actions=actions
    
    def __len__(self):
        return len(self.actions)
    
    def __iter__(self):
        return iter(self.actions)
    
    def map(self,fun):
        actions=[action_i.map(fun) 
                   for action_i in self.actions]
        return ActionGroup(actions)

    @classmethod
    def read(cls,in_path):
        actions=[ Action.read(path_i) 
                    for path_i in utils.top_files(in_path)]
        return cls(actions)

    def save(self,out_path):
        utils.make_dir(out_path)
        for action_i in self:#.actions:
            action_i.save(f"{out_path}/{action_i}")
   
    def unify(self,fun):
        all_items=[]
        for action_i in self.actions:
            all_items.extend(action_i.eval(fun))
        return all_items

    def split(self,fun=None):
        if(fun is None):
            fun= lambda desc: (desc.person % 2)==1
        train,test=[],[]
        for action_i in self:
            if(fun(action_i.desc)):
                train.append(action_i)
            else:
                test.append(action_i)
        return ActionGroup(train),ActionGroup(test)
    
    def as_dataset(self):
        X,y=[],[]
        for action_i in self:
            for frame_j in action_i:
                X.append(frame_j)
                y.append(action_i.desc.cat)
        return base.Dataset(X=np.array(X),
                            y=np.array(y))

    def stats(self,fun):
        values=self.unify(fun)
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
                   cat=int(raw[0]), 
                   person=int(raw[1]))

class Action(object):
    def __init__( self,
                 frames,
                 desc):
        self.frames = frames
        self.desc = desc
    
    def __len__(self):
    	return len(self.frames)

    def __str__(self):
        return self.desc.name
    
    def __iter__(self):
        return iter(self.frames)
    
    def eval(self,fun):
        return [ fun(frame_i) for frame_i in self]
    
    def map(self,fun):
        return Action(frames=self.eval(fun),
                      desc=self.desc )
    
    @classmethod
    def read(cls,in_path):
        frames=[cv2.imread(path_i,cv2.IMREAD_GRAYSCALE)
                    for path_i in utils.top_files(in_path)]
        desc=ActionDesc.from_path(in_path)
        return cls(frames,desc)

    def save(self,out_path):
        utils.make_dir(out_path)
        for i,frame_i in enumerate(self.frames):
            cv2.imwrite(f"{out_path}/{i}.png", frame_i)

def height(frame):
    return frame.shape[0]

def width(frame):
    return frame.shape[1]

if __name__ == '__main__':
    actions=ActionGroup.read("MSR(scaled)")

