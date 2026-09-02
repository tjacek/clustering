import numpy as np
import cv2
from collections import defaultdict
from dataclasses import dataclass
import argparse
from tqdm import tqdm
import base,utils

class SeqGroup(list):
    def __init__(self, actions=None):
        if(actions is None):
            actions=[]
        super().__init__(actions)
 
    @classmethod
    def dtype(cls):
        return Seq
    
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

    def map(self,fun):
        seqs=[seq_i.map(fun) 
                   for seq_i in tqdm(self)]
        return self.__class__(seqs)
    
    def map_seq(self,fun,group_type=None):
        if(group_type is None):
            group_type=self.__class__
        dtype=group_type.dtype()
        new_seqs=[ dtype( fun(seq_i),
                          seq_i.desc)
                     for seq_i in self]
        return group_type(new_seqs)
    
    def flatten(self,fun=None):
        if(fun is None):
            fun=identity
        all_items=[]
        for action_i in self:
            all_items.extend(action_i.eval(fun))
        return all_items
    
    def flatten_seq(self,fun=None):
        all_items=[]
        for seq_i in self:
            all_items.extend(fun(seq_i))
        return all_items

    def indexed_frames(self):
        get_index=GetIndex()
        def helper(frame):
            i=get_index()
            return (i,frame)
        pairs=self.flatten(helper)
        index,frames= zip(*pairs)
        return index,frames
           
    def indexed_map(self,fun):
        get_index=GetIndex()
        def helper(frame):
            i=get_index()
            return fun(i,frame)
        return self.map(helper)
        
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
        def helper(seq_i):
            cat=seq_i.desc.cat
            return [(frame_i,cat) for frame_i in seq_i]
        pairs=self.flatten_seq(helper)
        X,y=list(zip(*pairs))
        return base.Dataset(X=np.array(X),
                            y=np.array(y))
    def by_cat(self):
        cat_dict = defaultdict(lambda: self.__class__())
        for seq_i in self:
            cat_i=seq_i.desc.cat
            cat_dict[cat_i].append(seq_i)
        return cat_dict

    def as_dict(self):
        return { seq_i.desc.name:seq_i
                    for seq_i in self}

    def group(  self,
                label_group):
        frame_dict=defaultdict(list)
        info_dict=defaultdict(list)
        label_dict=label_group.as_dict()
        for seq_i in self:
            desc_i=seq_i.desc
            labeling_i=label_dict[desc_i.name]
            info_i=[desc_i.cat,desc_i.person]
            n_frames=len(seq_i)
            for j,frame_j in enumerate(seq_i):
                info_j= tuple([j/n_frames]+info_i)
                label_j=labeling_i[j]
                frame_dict[label_j].append(frame_j)
                info_dict[label_j].append(info_j)
        return frame_dict,info_dict

    def by_labels( self, 
                   label_group,
                   as_data=True):
        frame_dict,info_dict=self.group(label_group)
        data_dict={}
        if(as_data):
            for i,frames_i in frame_dict.items():
                info_i=info_dict[i]
                _,y_i,_=list(zip(*info_i))
                frames_i=np.array(frames_i)
                data_dict[i]=base.Dataset(frames_i,y_i)
        else:
            dtype=self.dtype()
            for i,frames_i in frame_dict.items():
                data_dict[i]=dtype( frames=frames_i,
                                    desc=ActionDesc(i))
        return data_dict

class _SeqGroup(list):
    def __init__(self, actions=None):
        if(actions is None):
            actions=[]
        super().__init__(actions)
    
    @classmethod
    def dtype(cls):
        return Seq
    
    def map(self,fun):
        seqs=[seq_i.map(fun) 
                   for seq_i in tqdm(self)]
        return self.__class__(seqs)
    


    def eval(self,fun,group_type):
        dtype=group_type.dtype()
        raw_values= [ dtype(frames=seq_i.eval(fun),
                            desc=seq_i.desc)
                        for seq_i in self]
        return group_type(raw_values)



    def save(self,out_path):
        utils.make_dir(out_path)
        for seq_i in self:
            seq_i.save(f"{out_path}/{seq_i}")
       
    @classmethod
    def _from_actions( cls, action_path, model, fun):
        actions = ActionGroup.read(action_path)
        seqs = cls([])
        for action_i in tqdm(actions):
            X = np.array(action_i)
            seq_i = cls.dtype()(frames=fun(X), desc=action_i.desc)
            seqs.append(seq_i)
        return seqs



    def map_with_index(self,fun,dtype=None):
        get_index=GetIndex()
        def helper(seq_i):
            return [fun(get_index(),frame_j,seq_i.desc)
                     for frame_j in seq_i]     
        return self.map_seq(helper,dtype)

        def helper(x):
            return fun(get_index(),x)
        return self.map(helper)

    def lazy_save(self,fun,out_path):
        utils.make_dir(out_path)
        for seq_i in tqdm(self):
            new_seq_i=seq_i.map(fun)
            new_seq_i.save(f"{out_path}/{seq_i}")

    def cats(self):
        def helper(seq_i):
            return [seq_i.desc.cat for _ in  seq_i]
        return self.flatten_seq(helper)


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
    cat:int = 0
    person:int = 0  
    
    @classmethod
    def from_path(cls,path):
        name=path.split("/")[-1]
        return cls.from_name(name)

    @classmethod
    def from_name(cls,name):
        name=name.split(".")[0]
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
    
    def mean_img(self,out_path=None):
        if(out_path):
            utils.make_dir(out_path)
        for action_i in self:
            out_i=f"{out_path}/{action_i}"
            action_i.mean_img(out_i)

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

    def mean_img(self,out_path=None):
        X=np.array(self)
        mean_img=np.mean(X,axis=0)
        if(out_path):
            cv2.imwrite(f"{out_path}.png", mean_img)
        return mean_img

def height(frame):
    return frame.shape[0]

def width(frame):
    return frame.shape[1]

def identity(x):
    return x

class GetIndex(object):
    def __init__(self):
        self.counter=0
    
    def __call__(self,frame=None):
        old_value=self.counter
        self.counter+=1
        return old_value

class Proj(object):
    def __init__(self,k=0):
        self.k=k
    
    def __call__(self,arg_tuple):
        return arg_tuple[self.k]
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str,default="MSR/raw")
    parser.add_argument("--out_path", type=str,default="MSR/scaled")
    args=parser.parse_args()
    actions=ActionGroup.read(args.in_path)
    actions=actions.rescale()
    actions.save(args.out_path)