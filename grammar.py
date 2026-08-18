import numpy as np
import sksequitur
from sksequitur import Grammar, Mark, Parser, parse
from nltk import PCFG
import string
import argparse
import labels,utils

class GrammarAdapter(object):
    def __init__(self,grammar):
        self.grammar=grammar
        self._letter_dict=None
    
    @property
    def letter_dict(self):
        if(self._letter_dict is None):
            self._letter_dict={}
            n=len(string.ascii_uppercase)
            for key_i,rule_i in self.grammar.items():
                key_i=int(key_i)
                if(key_i==0):
                    continue
                k= int(np.floor(key_i/n))
                nonterm_i=string.ascii_uppercase[key_i%n]
                nonterm_i+=f"_{k}"
                self._letter_dict[key_i]=nonterm_i
        return self._letter_dict

    @classmethod
    def from_symb(cls,symb_dict):
        parser_i = Parser()
        for symb_j in symb_dict.values():
            parser_i.feed(symb_j)
            parser_i.feed([Mark()])
        return cls(Grammar(parser_i.tree))
    
    def print(self):
        print(len(self.grammar))
        for key_i,value_i in self.grammar.items():
            print(f"{key_i}->{value_i}")

    def as_string(self):
        rules=[]
        def helper(elem_j):
            if(isinstance(elem_j,int)):
                return self.letter_dict[elem_j]
            if(type(elem_j)==Mark):
                return "|"
            else:
                return f"'{elem_j}'"
        for key_i,rule_i in self.grammar.items():
            left_i=" ".join([helper(elem_j) 
                                for elem_j in rule_i])
            left_i=left_i.split("|")
            if(len(left_i)>1):
                prob_i=f"[{1.0/len(left_i):.4f}]"
                left_i=[ f"{elem_j} {prob_i} " 
                           for elem_j in left_i 
                               if(len(elem_j)>0)]
                left_i="|".join(left_i)
            else:
                left_i=left_i[0]
                left_i=f"{left_i} [1.0]"
            if(key_i==0):
                right_i="START"
            else:
                right_i=self.letter_dict[key_i]
            rule_i=f"{right_i} -> {left_i}"
            rules.append(rule_i)
        return "\n".join(rules)            

def build_grammars(cls_path):
    labeling= labels.LabelingGroup.read(cls_path)
    labeling=compress(labeling)
    dict_map=labels.DictMap.tf_map(labeling)
    train,test=labeling.split()
    for cat_i,group_i in train.by_cat().items():
        print(f"Classs:{cat_i}")
        symb_dict=group_i.as_symbols( symb_map=dict_map,
                                      verbose=False)
        grammar_i= GrammarAdapter.from_symb(symb_dict)
        print(grammar_i.as_string())
#        print(list(grammar_i.values())[0])
#        print("**************************")

def show_symbols(cls_path):
    print(cls_path)
    labeling= labels.LabelingGroup.read(cls_path)
    labeling=compress(labeling)
    symb_dict=labeling.as_symbols("tf-idf",False)
    utils.print_dict(symb_dict)

def compress(labeling,max_coe=4):
    def helper(seq_i):
        count=0
        current=seq_i[0]
        new_seq=[current]
        max_i=max_coe-1
        for frame_j in seq_i[1:]:
            if(current!=frame_j):
                new_seq.append(frame_j)
                count=0
                current=frame_j
                continue
            if(current==frame_j and  count<max_i):
                count+=1
                new_seq.append(frame_j)
        return new_seq
    return labeling.map_seq(helper)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--cmd", type=str,default="build")
    args=parser.parse_args()
    if(args.cmd=="build"):
        build_grammars(args.cls_path)
    if(args.cmd=="show"):
        show_symbols(args.cls_path)