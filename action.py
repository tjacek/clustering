import numpy as np
import cv2
import utils

class ActionGroup(object):
    def __init__(self, actions):
        self.actions=actions
    
    def __len__(self):
        return len(self.actions)
    
    @classmethod
    def read(cls,in_path):
        actions=[ Action.read(path_i) 
                    for path_i in utils.top_files(in_path)]
        return cls(actions)
    
    def  max_dims(self):
         return [action_i.max_dims() for action_i in self.actions] 

class Action(object):
    def __init__(self,frames):
        self.frames=frames
    
    def __len__(self):
    	return len(self.frames)
    
    @classmethod
    def read(cls,in_path):
        frames=[cv2.imread(path_i,cv2.IMREAD_GRAYSCALE)
                    for path_i in utils.top_files(in_path)]
        return cls(frames)

    def shape(self):
    	return [frame_i.shape for frame_i in self.frames]

    def max_dims(self):
    	arr=np.array(self.shape())
    	print(arr)
    	return np.amax(arr,axis=0)

actions=ActionGroup.read("MSR")
print(actions.max_dims())