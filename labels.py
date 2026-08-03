import numpy as np
import string
from collections import defaultdict
from dataclasses import dataclass
import seq,cnn,utils


class FeatSeqGroup(seq.SeqGroup):
    @classmethod
    def dtype(cls):
        return FeatSeq

    @classmethod
    def from_actions(cls, action_path, model_path, n_layer=1):
        return cls._from_actions(action_path, model_path,
                                  lambda model, X: model.extract(X, n_layer))

    def as_precluster(self):
        return Preclustering.from_feats(self)

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
    
    def hist(self,n_clusters=None):
        raw=np.array(self.flatten())
        if(n_clusters is None):
            n_clusters=np.amax(raw)+1
        hist=np.zeros((n_clusters,))
        for label_i in raw:
            hist[label_i]+=1
        return hist
#        return np.histogram(raw,bins=n[0]

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
        return np.array(list(set(labels)))

    def as_symbols( self,
                    symb_map=None,
                    verbose=True):
        if(symb_map is None):
            symb_map=BasicMap()
        if(symb_map=="tf-idf"):
            symb_map=DictMap.tf_map(self)
        symb_dict={ str(seq_i): seq_i.as_symbols(symb_map) 
                    for seq_i in self}
        if(verbose):
            utils.print_dict(symb_dict)
        return symb_dict

    def tf_idf(self):
        full_hist = self.hist()
        n_clusters=full_hist.shape[0]
        cat_dict = self.by_cat()
        n_cats=len(cat_dict)
        def relative_freq(i):
            hist_i=cat_dict[i].hist(n_clusters)
            return [cat_j / full_j
                    for cat_j, full_j in zip(hist_i,full_hist)]
        tf_arr= np.array([relative_freq(i) 
                        for i in range(n_cats)])
        return tf_arr

class Labeling(seq.Seq):
    @classmethod
    def read(cls,in_path):
        arr=np.loadtxt(in_path).astype(int)
        desc=seq.ActionDesc.from_path(in_path)
        return cls(arr,desc)

    def save(self,out_path):
        np.savetxt(out_path,self)
    
    def unique(self):
        return list(set(self))

    def as_symbols(self,symb_map):
        return [ symb_map(label) 
                  for label in self]

@dataclass
class Preclustering:
    frames:np.ndarray
    indexes:np.ndarray
    order_labeling:LabelingGroup
    
    @classmethod
    def from_feats(cls,feat_seqs):
        pairs=feat_seqs.map_with_index(lambda i,x:(i,x))
        pairs=pairs.flatten()
        indexes,frames=zip(*pairs)
        return cls( np.array(frames),
                    np.array(indexes),
                    feat_seqs.eval(seq.GetIndex(),
                                   LabelingGroup))

class BasicMap(object):
    def __init__(self):
        self.letters = list(string.ascii_lowercase)
        self.n=len(self.letters)
    
    def to_symb(self,i):
        return self.letters[i % self.n]
    
    def __call__(self,label):
        symb=self.to_symb(label)
        k= int(np.floor(label/self.n))
        if(k==0):
            return symb
        return f"{symb}_{k}"

class DictMap(object):
    def __init__(self,symb_dict):
        self.symb_dict=symb_dict

    def __call__(self,label):
        return self.symb_dict[label]

    @classmethod
    def tf_map(cls,label_group):
        tf_arr=label_group.tf_idf()
        raise Exception(tf_arr)

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

if __name__ == '__main__':
#train("MSR/scaled","MSR/model")
#seqs=FeatSeqGroup.from_actions("MSR/scaled","MSR/model.keras")
#seqs.save("MSR/seq")
    seqs= FeatSeqGroup.read("MSR/seq")
