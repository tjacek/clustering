import numpy as np
import string
import action,cnn,utils

class FeatSeqGroup(action.SeqGroup):
    @classmethod
    def dtype(cls):
        return FeatSeq

    @classmethod
    def from_actions(cls, action_path, model_path, n_layer=1):
        return cls._from_actions(action_path, model_path,
                                  lambda model, X: model.extract(X, n_layer))

class FeatSeq(action.Seq):
    def save(self,out_path):
        np.savetxt(out_path,self)

class LabelingGroup(action.SeqGroup):
    @classmethod
    def dtype(cls):
        return Labeling

    @classmethod
    def from_actions(cls, action_path, model_path):
        return cls._from_actions(action_path, model_path,
                                  lambda model, X: model.predict(X))

    def unique(self):
        labels=[]
        for seq_i in self:
            labels+=seq_i.unique()
        labels=list(set(labels))
        return np.array(labels)

class Labeling(action.Seq):
    def save(self,out_path):
        np.savetxt(out_path,self)
    
    def unique(self):
        return list(set(self))

    def as_symbols(self):
        letters = list(string.ascii_lowercase)
        n=len(letters)
        symb_seq=""
        def helper(i):
            symb=letters[i%n]
            k= int(np.floor(i/n))
            return symb+str(k)
        for i in self:
            symb_seq+=helper(i)
        return symb_seq

def train( in_path,
           out_path,
           epochs=150):
    actions=action.ActionGroup.read(in_path)
    train,test=actions.split()
    model,_=cnn.cnn_exp( train.as_dataset(),
                         test.as_dataset(),
                         cnn.frame_params(),
                         epochs=epochs)
    model.save(out_path)

#train("MSR/scaled","MSR/model")
#seqs=FeatSeqGroup.from_actions("MSR/scaled","MSR/model.keras")
seqs=LabelingGroup.from_actions("MSR/scaled","MSR/model.keras")
cat_dict=seqs.by_cat()
for cat_i,group_i in cat_dict.items():
    print(cat_i)
    print(group_i.unique())
#seqs.save("MSR/seqs")