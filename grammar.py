import numpy as np
import sksequitur
from sksequitur import Grammar, Mark, Parser, parse
import nltk
from nltk.parse import ViterbiParser,InsideChartParser
from nltk.parse import ChartParser
from collections import Counter
#from nltk import PCFG
import string
import argparse
import seq
import utils

class GrammarEnsemble(dict):
    def pred_prob(self,sentence):
        keys=list(self.keys())
        keys.sort()
        prob_vector=[]
        for key_i in keys:
            if(self.lack_terminals(key_i,sentence)):
                prob_vector.append(0)
                continue
            grammar_i=self[key_i]
            prob_i=viterbi_alg(grammar_i,sentence)
            prob_vector.append(prob_i)
        prob_vector = np.array(prob_vector)
        return prob_vector/np.sum(prob_vector)
        
    def pred(self,sentence):
        prob=self.pred_prob(sentence)
        return np.argmax(prob)

    def lack_terminals(self,key_i,sentence):
        grammar_i=self[key_i]
        raise Exception(dir(grammar_i))
        covered = set(grammar_i._lexical_index.keys()) 
        for term_j in sentence:
            if(not term_j in covered):
                return True
        return False
    
    def count(self,key_i,pos_sent):
        grammar_i=self[key_i]
        parser = ChartParser(grammar_i)
        rule_counts = Counter()
        for sentence in pos_sent:
            trees = list(parser.parse(sentence))
            if not trees:
               continue
            tree = trees[0]
        for production in tree.productions():
            rule_counts[production] += 1
        print("Liczności reguł:")
        for rule, count in rule_counts.items():
            print(count, rule)

def viterbi_alg(grammar_i,sentence):
    parser_i=ViterbiParser(grammar_i)            
    prob_i=[ tree.prob() for tree in parser_i.parse(sentence)]
    if(len(prob_i)==0):
        return 0
    else:
        return np.amax(prob_i)

def inside_alg(grammar_i,sentence):
    parser_i=ViterbiParser(grammar_i)            
    total_prob = 0.0
    for tree in parser_i.parse(sentence):
#        print(tree, "->", tree.prob())
        total_prob += tree.prob()
    return total_prob

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
                           for elem_j in left_i] 
#                               if(len(elem_j)>0)]
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
    label_group=seq.get_group("labels")
    labeling= label_group.read(cls_path)
    labeling=compress(labeling)
    symbols=labeling.as_symbols()
    train,test=symbols.split()
    grammar_ens=GrammarEnsemble()
    for cat_i,group_i in symbols.by_cat().items():
        print(f"Classs:{cat_i}")
    raise Exception(len(symbols))
#    dict_map=labels.DictMap.tf_map(labeling)
#    train,test=labeling.split()
#    grammar_ens=GrammarEnsemble()
    for cat_i,group_i in train.by_cat().items():
#        print(f"Classs:{cat_i}")
#        symb_dict=group_i.as_symbols( symb_map=dict_map,
#                                      verbose=False)
        grammar_i= GrammarAdapter.from_symb(symb_dict)
#        raise Exception(dir(grammar_i.grammar))
        str_grammar_i=grammar_i.as_string()
        prob_gram_i=nltk.PCFG.fromstring(str_grammar_i)
        grammar_ens[cat_i]=prob_gram_i
        grammar_ens.count( cat_i,
                           list(symb_dict.values()))
    symb_test=train.as_symbols(symb_map=dict_map,
                               verbose=False)
    hard_predic(grammar_ens,symb_test)

def hard_predic(grammar_ens,symb_test):
    error=[]
    for name_i,symb_i in symb_test.items():
        pred_cat=grammar_ens.pred(symb_i)
        desc_i=seq.ActionDesc.from_name(name_i)
        error.append(int(desc_i.cat==pred_cat))
    print(np.mean(error))

def soft_predic(grammar_ens,symb_test):
    n_cats=len(grammar_ens)
    total_error=0
    for name_i,symb_i in symb_test.items():
        desc_i=seq.ActionDesc.from_name(name_i)
        one_hot=np.zeros((n_cats,))
        one_hot[desc_i.cat]=1
        pred_cat=grammar_ens.pred_prob(symb_i)
        err_i= np.sum(np.abs(pred_cat-one_hot))
        total_error+=err_i
    print(total_error/len(symb_test))

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