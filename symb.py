import numpy as np
from scipy.stats import entropy
from itertools import combinations  
import argparse
import grammar.core
import seq
import plot
from grammar.markov import MarkovChain

def markov(cls_path):
    for cat_i,markov_i in markov_iter(cls_path):
        arr=markov_i.as_matrix()
        plot.show_heatmap(arr,
                          title=cat_i,
                          x_axis=markov_i.states,
                          y_axis=markov_i.terms)

def markov_iter(cls_path):
    label_group=seq.get_group("labels")
    labeling= label_group.read(cls_path)
    symbols=labeling.as_symbols()
    for cat_i,group_i in symbols.by_cat().items():
        yield cat_i,MarkovChain.make(group_i)

def diver(cls_path,verbose=True):
    def helper(group,states):
        markov=MarkovChain.make(group)
        dist= markov.stationary_dist( states=states,
                                      terms=states)
        return dist
    cats,entr=[],[]
    for i,(train_i,test_i) in diver_iter(helper,cls_path):
        if(verbose):
            print(i)
            print(train_i)
            print(test_i)
        entr_i=entropy( train_i,
                        test_i)
        cats.append(i)
        entr.append(entr_i)
    entr=np.array(entr)
    indices=np.argsort(entr)
    print(entr)
    print(indices)

def diver_iter(fun,cls_path):
    label_group=seq.get_group("labels")
    labeling= label_group.read(cls_path)
    symbols=labeling.as_symbols()
    for cat_i,group_i in symbols.by_cat().items():
        states=group_i.unique()
        train_i,test_i=group_i.split()
        train_value=fun(train_i,states)
        test_value=fun(test_i,states)
        yield cat_i,(train_value,test_value)

def diver_cat(cls_path,verbose=True):
    markovs=list(markov_iter(cls_path))
    n=len(markovs)
    matrix=np.zeros((n,n))

    pairs=list(combinations(markovs, 2))
    for (i,markov_i),(j,markov_j) in pairs:
        states= markov_i.states + markov_i.states
        states=list(set(states))
        if(i!=j):
            matrix[i][j]= markov_j.kl_divergence(markov_i,states)
        matrix[j][i]= markov_i.kl_divergence(markov_j,states)
    plot.show_heatmap(matrix,title="kl_divergence")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--cmd", type=str,default="diver_cat")
    args=parser.parse_args()
    if(args.cmd=="build"):
        grammar.core.build_grammars(args.cls_path)
    if(args.cmd=="markov"):
    	markov(args.cls_path)
    if(args.cmd=="diver"):
        diver(args.cls_path)
    if(args.cmd=="diver_cat"):
        diver_cat(args.cls_path)