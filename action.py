import numpy as np
import cv2
from dataclasses import dataclass
import utils

class ActionGroup(object):
    def __init__(self, actions):
        self.actions=actions
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self,i):
        return self.actions[i]

    @classmethod
    def read(cls,in_path):
        actions=[ Action.read(path_i) 
                    for path_i in utils.top_files(in_path)]
        return cls(actions)
    
    def  max_dims(self):
         return [action_i.max_dims() for action_i in self.actions] 


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
        self.frames=frames
        self.desc=desc
    
    def __len__(self):
    	return len(self.frames)

    def __str__(self):
        return self.desc.name
    
    @classmethod
    def read(cls,in_path):
        frames=[cv2.imread(path_i,cv2.IMREAD_GRAYSCALE)
                    for path_i in utils.top_files(in_path)]
        desc=ActionDesc.from_path(in_path)
        return cls(frames,desc)

    def shape(self):
    	return [frame_i.shape for frame_i in self.frames]

    def max_dims(self):
    	arr=np.array(self.shape())
    	print(arr)
    	return np.amax(arr,axis=0)

actions=ActionGroup.read("MSR")
print(f"{actions[0]}")