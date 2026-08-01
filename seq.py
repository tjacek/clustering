import numpy as np
import string
import action,cnn,utils

class FeatSeqGroup(action.SeqGroup):
    @classmethod
    def dtype(cls):
        return FeatSeq

    @classmethod
    def from_actions( cls,
                      action_path,
                      model_path,
                      n_layer=1):
        actions=action.ActionGroup.read(action_path)
        model=cnn.ConvNN.read(model_path)
        seqs=cls([])
        for action_i in actions:
            X=np.array(action_i)
            seq_i=FeatSeq( frames=model.extract(X,n_layer),
                           desc=action_i.desc)
            seqs.append(seq_i)
        return seqs

class FeatSeq(action.Seq):
    def save(self,out_path):
        np.savetxt(out_path,self)

class LabelingGroup(action.SeqGroup):
    @classmethod
    def dtype(cls):
        return Labeling

    @classmethod
    def from_actions( cls,
                      action_path,
                      model_path):
        actions=action.ActionGroup.read(action_path)
        model=cnn.ConvNN.read(model_path)
        seqs=cls([])
        for action_i in actions:
            X=np.array(action_i)
            seq_i=Labeling( frames=model.predict(X),
                            desc=action_i.desc)
            print(seq_i.as_symbols())
            seqs.append(seq_i)
        return seqs

class Labeling(action.Seq):
    def save(self,out_path):
        np.savetxt(out_path,self)

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
#seqs.save("MSR/seqs")