import numpy as np
import reduct
import argparse
import seq
import plot
import utils

class GroupedClust(object):
    def __init__( self,
                  by_labels):
        self.by_labels=by_labels

    def info_types(self):
#        dtype=self.by_labels.dtype()
        return ["y","order","person"]

    def names(self):
        return list(self.by_labels.keys())
    
    def __iter__(self):
        return iter(self.by_labels.items())
    
    def __getitem__(self,item):
        return self.by_labels[item]
    
    def n_frames(self):
        sizes=[len(data_i) for i,data_i in self]
        return sum(sizes)

    @classmethod
    def make( cls,
              label_path,
              seq_path):
        label_group=seq.get_group("labels")
        labeling= label_group.read(label_path)
        symbols=labeling.as_symbols(symb_map="tf-idf")
        seq_group=seq.get_group("feat")
        seqs= seq_group.read(seq_path)
        by_labels=seqs.group_info(symbols)
        return cls(by_labels)

def cluster_stats( label_path,
	               seq_path):
    by_labels=GroupedClust.make(label_path,seq_path)
    print(by_labels.n_frames())
    for label_i,data_i in by_labels:
        print(f"Cat:{label_i}")
        X_tsne=reduct.umap_reduct(data_i)
        plot.adno_plot(x=X_tsne[:,0],
                       y=X_tsne[:,1],
                       label=data_i.y,
                       title=label_i)


def split_frames( action_path,
                  cls_path,
                  out_path="out"):
    action_group=seq.get_group("actions")
    actions=action_group.read(action_path)
    label_group=seq.get_group("labels")
    labeling= label_group.read(cls_path)
    symbols=labeling.as_symbols(symb_map="tf-idf")
    by_labels=actions.by_labels(symbols,as_data=False)
    utils.make_dir(out_path)
    for i,cls_i in by_labels.items():
        cls_i.save(f"{out_path}/{i}")
#        print(i)
#        print(type(cls_i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--seq_path", type=str,default="MSR/ae/layer_1/seqs")
    parser.add_argument("--action_path", type=str,default="MSR/scaled")
    parser.add_argument("--cmd", type=str,default="split")
    args=parser.parse_args()
    if(args.cmd=="stats"):
        cluster_stats( args.cls_path,
	                   args.seq_path)
    if(args.cmd=="split"):
        split_frames( args.action_path,
                      args.cls_path)