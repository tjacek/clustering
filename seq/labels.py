import numpy as np
import string
from collections import defaultdict
from dataclasses import dataclass
import seq.core
import utils

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

class LabelingGroup(seq.core.SeqGroup):
    @classmethod
    def dtype(cls):
        return Labeling

    @classmethod
    def from_actions(cls, action_path, model):
        return cls._from_actions(action_path, model,
                                  lambda X: model.predict(X))
    
    def hist(self,n_clusters=None):
        raw=np.array(self.flatten())
        if(n_clusters is None):
            n_clusters=np.amax(raw)+1
        return hist_fun(raw,n_clusters)

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

    def n_clust(self):
        return np.amax(self.unique())+1
    
    def unique(self):
        labels=[]
        for seq_i in self:
            labels+=seq_i.unique()
        return np.array(list(set(labels)))

    def as_symbols( self,
                    symb_map="tf-idf"):
#                    verbose=True):
        if(symb_map is None):
            symb_map=BasicMap()
        if(symb_map=="tf-idf"):
            symb_map=DictMap.tf_map(self)
        symb_seq=[]
        for seq_i in self:
            desc_i=seq_i.desc
            values_i=seq_i.as_symbols(symb_map)
            symb_i=SymbolSeq(values_i,desc_i)
            symb_seq.append(SymbolSeq(symb_i,desc_i))
        return SymbGroup(symb_seq)

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

class Labeling(seq.core.Seq):
    @classmethod
    def read(cls,in_path):
        arr=np.loadtxt(in_path).astype(int)
        desc=seq.core.ActionDesc.from_path(in_path)
        return cls(arr,desc)
    
    def save(self,out_path):
        np.savetxt( out_path,self, fmt='%i')
    
    def unique(self):
        return list(set(self))

    def as_symbols(self,symb_map=None):
        return [ symb_map(label) 
                  for label in self]
    
    def as_numpy(self):
        return np.array(self,dtype=int)
    
    def hist(self,n_clusters=None):
        return hist_fun(self,n_clusters)


class SymbGroup(seq.core.SeqGroup):
    SEP="|"
    @classmethod
    def dtype(cls):
        return SymbolSeq

    def print(self):
        for seq_i in self:
            print(seq_i.desc)
            print(list(seq_i))

    def ngram_dict(self,k):
        ngram_dict=defaultdict(lambda :0)
        for seq_i in self:
            for ngram_j in seq_i.ngrams(k):
                ngram_id=self.SEP.join(ngram_j)
                ngram_dict[ngram_id]+=1
        return ngram_dict
    
    def unique(self):
        labels=[]
        for seq_i in self:
            labels+=seq_i.unique()
        return np.array(list(set(labels)))

class SymbolSeq(seq.core.Seq):
    def bigrams(self):
        n=len(self)-1
        return [ (self[i],self[i+1]) for i in range(n)]

    def ngrams(self,k):
        n=len(self)-k
        return [ (self[i:i+k]) for i in range(n)]
    
    def unique(self):
        return list(set(self))

def hist_fun(arr,n_clusters):
    hist=np.zeros((n_clusters,))
    for label_i in arr:
        hist[label_i]+=1
    return hist

@dataclass
class Preclustering:
    frames:np.ndarray
    indexes:np.ndarray
    cats:np.ndarray
    order_labeling:LabelingGroup
    
    @classmethod
    def from_feats(cls,feat_seqs):
        def helper(i,frames_i,desc):
            return (i,frames_i,desc.cat)
        tuple_seqs=feat_seqs.map_with_index(helper,
                                            seq.core.SeqGroup)
        tuples=tuple_seqs.flatten()
        indexes,frames,cats=zip(*tuples)
        return cls( np.array(frames),
                    np.array(indexes),
                    np.array(cats),
                    tuple_seqs.eval(seq.core.Proj(0),
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
        cat_dict=defaultdict(lambda:[])
        tf_arr=label_group.tf_idf()
        for i,cls_i in enumerate(tf_arr.T):
            k=np.argmax(cls_i)
            value=cls_i[k]
            cat_dict[k].append((i,value))
        letters=string.ascii_letters
        cls_dict={}
        for cat_i,pairs_i in cat_dict.items():
            cls_i,score_i=zip(*pairs_i)
            score_i=np.array(score_i)
            symb_i=str(cat_i+1) #letters[cat_i]
            indexes=np.argsort(score_i)[::-1]
            for j in indexes:
                cls_j=cls_i[j]
                cls_dict[cls_j]=f"{symb_i}_{letters[j]}"
        return cls(cls_dict)

class SymbDict(dict):
    def bigrams(self):
        bg_dict= defaultdict(lambda:0)
        for symb_i in self.values():
            for a,b in zip(symb_i, symb_i[1:]):
                pair_ij=f"{a}-{b}"
                bg_dict[pair_ij]+=1
        return bg_dict

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