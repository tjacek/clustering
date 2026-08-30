import numpy as np
from sklearn import manifold
import argparse
import seq
import plot

def cluster_stats( label_path,
	               seq_path):
    label_group=seq.get_group("labels")
    labeling= label_group.read(label_path)
    symbols=labeling.as_symbols(symb_map="tf-idf")
    seq_group=seq.get_group("feat")
    seqs= seq_group.read(seq_path)
    by_labels=seqs.by_labels(symbols)
    for label_i,data_i in by_labels.items():
        print(f"Cat:{label_i}")
        t_sne = manifold.TSNE( n_components=2,
                               perplexity=min(30,len(data_i)-1),
                               init="random",
                               max_iter=250,
                               random_state=0,
                              )
        X_tsne = t_sne.fit_transform(data_i.X)
        plot.adno_plot(x=X_tsne[:,0],
                       y=X_tsne[:,1],
                       label=data_i.y,
                       title=label_i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_path", type=str,default="MSR/spectral_36")
    parser.add_argument("--seq_path", type=str,default="MSR/seqs")
    args=parser.parse_args()
    cluster_stats( args.cls_path,
	               args.seq_path)