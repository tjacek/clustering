import numpy as np
from scipy.stats import entropy
import grammar.core
import seq
import argparse
from grammar.markov import MarkovChain
import plot

def markov(cls_path):
    label_group=seq.get_group("labels")
    labeling= label_group.read(cls_path)
    symbols=labeling.as_symbols()
    for cat_i,group_i in symbols.by_cat().items():
        dist_i=MarkovChain.make(group_i)
        arr=dist_i.as_matrix()
        plot.show_heatmap(arr,
                          title=cat_i,
                          x_axis=dist_i.states,
                          y_axis=dist_i.terms)

def diver(cls_path):
    label_group=seq.get_group("labels")
    labeling= label_group.read(cls_path)
    symbols=labeling.as_symbols()
    for cat_i,group_i in symbols.by_cat().items():
        train_i,test_i=group_i.split()
        train_dist_i=train_i.ngram_dict(1)
        test_dist_i=test_i.ngram_dict(1)
        terms=group_i.unique()
        train_arr=to_array(terms,train_dist_i)
        test_arr=to_array(terms,test_dist_i)
        print(cat_i)
        eps=np.ones(test_arr.shape)/(10**6)
        print(entropy(train_arr+eps,
                      test_arr+eps))
        print(train_arr)
        print(test_arr)

def to_array(terms,dist):
    arr=[]
    for term_i in terms:
        if(term_i in dist):
            arr.append(dist[term_i])
        else:
            arr.append(0)
    arr=np.array(arr,dtype=float)
    arr/=np.sum(arr)
    return arr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--cmd", type=str,default="diver")
    args=parser.parse_args()
    if(args.cmd=="build"):
        grammar.core.build_grammars(args.cls_path)
    if(args.cmd=="markov"):
    	markov(args.cls_path)
    if(args.cmd=="diver"):
        diver(args.cls_path)
