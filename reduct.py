import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn import manifold
from sklearn.decomposition import PCA
import umap
import argparse
import seq
import plot
import utils

def reduce_dim( in_path,
                out_path,
                n_clusters):
    feat_group=seq.get_group("feat")
    feat_seqs=feat_group.read(in_path)
    indexes,frames=feat_seqs.indexed_frames()
    reduced_frames=spectral_reduct( frames,
                                    n_components=2)
    def helper(i,frame):
        return reduced_frames[i]
    reduced_seqs= feat_seqs.indexed_map(helper)
    reduced_seqs.save(out_path)

def make_plot(in_path,out_path):
    feat_seqs=seq.get_group("feat").read(in_path)
    utils.make_dir(out_path)
    for seq_i in feat_seqs:
        dist_i=seq_i.distance()
        cum_i=np.cumsum(dist_i)
        plot.scatter( x=None, 
                      y=cum_i, 
                      title=str(seq_i),
                      xlabel="t",
                      ylabel="distance",
                      out_path=out_path)

def spectral_reduct(X,n_components=2):
    spectral = SpectralEmbedding( n_components=n_components, 
                                  n_neighbors=10, 
                                  random_state=42)
    return spectral.fit_transform(frames)

def pca_reduct(X):
    reducer = PCA(n_components=2)
    return reducer.fit_transform(X)

def umap_reduct(X):
    reducer = umap.UMAP( n_components=2,
                         random_state=0)
    return reducer.fit_transform(X)

def tsne_reduct(X):
    t_sne = manifold.TSNE( n_components=2,
                           perplexity=min(30,len(X)-1),
                           init="random",
                           max_iter=250,
                           random_state=0,
                        )
    return t_sne.fit_transform(X)

ALGS={ "spectral":spectral_reduct,
       "pca":pca_reduct,
       "umap":umap_reduct,
       "tsne":tsne_reduct}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str,default="MSR/ae/layer_1/seqs")
    parser.add_argument("--out_path", type=str,default="MSR/ae/layer_1/seqs")
    parser.add_argument("--n_clusters", type=int,default=46)
    parser.add_argument("--plot_path", type=str,default="plots")
    parser.add_argument("--cmd", type=str,default="plot")
    args=parser.parse_args()
    if(args.cmd=="compute"):
        reduce_dim( args.in_path,
                    args.out_path,
                    args.n_clusters)
    if(args.cmd=="plot"):
        make_plot(args.out_path,
                  args.plot_path)