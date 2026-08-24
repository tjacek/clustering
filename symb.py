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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--cmd", type=str,default="markov")
    args=parser.parse_args()
    if(args.cmd=="build"):
        grammar.core.build_grammars(args.cls_path)
    if(args.cmd=="markov"):
    	markov(args.cls_path)
