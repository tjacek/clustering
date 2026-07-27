import cv2
import utils

class Action(object):
    def __init__(self,frames):
        self.frames=frames
    
    def __len__(self):
    	return len(self.frames)
    @classmethod
    def read(cls,in_path):
        frames=[cv2.imread(path_i)
                    for id_i,path_i in utils.iter_files(in_path)]
        return Action(frames)

action=Action.read("MSR/1_1_1")
print(len(action))