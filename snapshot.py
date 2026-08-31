import numpy as np
import reduct
import argparse
import seq
import plot

class GroupedClust(object):
    def __init__( self,
                  by_labels):
        self.by_labels=by_labels

    def names(self):
        return list(self.by_labels.keys())
    
    def __iter__(self):
        return iter(self.by_labels.items())
    
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
        by_labels=seqs.by_labels(symbols)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/ae/layer_1/spectral_36")
    parser.add_argument("--seq_path", type=str,default="MSR/ae/layer_1/seqs")
    args=parser.parse_args()
    cluster_stats( args.cls_path,
	               args.seq_path)