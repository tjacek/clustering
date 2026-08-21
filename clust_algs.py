import numpy as np
from tqdm import tqdm
import argparse
import clusters
import seq
import plot
import utils

class LayerDir(object):
    def __init__(self,path):
        self.path=path
        self._seqs=None
        self._frames=None
        self._cats=None

    def labelings(self,alg_type):
        regex= rf"{alg_type}_\d+"
        return utils.find_paths(layer_path,regex )

    @property
    def seqs(self):
        if(self._seqs is None):
            label_group=seq.get_group("feat")
            self._seqs=label_group.read(f"{self.path}/seqs")
        return self._seqs
    
    @property
    def frames(self):
        if(self._frames is None):
            self._frames=np.array(self.seqs.flatten())
        return self._frames

    @property
    def cats(self): 
        if(self._cats is None):
            fun= lambda seq_i:seq_i.desc.cat
            self._cats= self.seqs.cats()
        return self._cats

def make_clust( seqs,
                layer_path,
                n_clusters=None,
                alg_type="spectral"):

    if(alg_type=="spectral"):   
        train,test=seqs,seqs
    else:
         train,test=seqs.split()
    alg=clusters.get_cluster_alg(alg_type)
    precluster=train.as_precluster()
    if( type(n_clusters)==int):
        n_clusters=[n_clusters]
    for k in tqdm(n_clusters):
        if(k<2):
            continue
        assig=alg(precluster,k)
        cls_labels=assig.get_labels(seqs)
        clust_name=f"{alg_type}_{k}"
        cls_labels.save(f"{layer_path}/{clust_name}")

def eval_clust( layer_dir,
                alg_type="kmeans",
                score_type="adj_mutual"):
    score_fun=clusters.get_score(score_type)
    label_group=seq.get_group("labels")
    scores,sizes=[],[]
    for path_i in tqdm(layer_dir.labelings(alg_type)):
        labeling_i=label_group.read(path_i)
        labels_i=labeling_i.flatten()
        score_i=score_fun( layer_dir,
                           labels_i)
        scores.append(score_i)
        sizes.append(len(scores)+1)
    plot.scatter( sizes, scores, 
                  title=alg_type,
                  xlabel="n_clusters",
                  ylabel=score_type)    
    scores=np.array(scores)
    print(scores)
    best=np.argmax(scores)
    print(f"Best Clusters:{sizes[best]}")
    print(f"Score:{scores[best]:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn_path", type=str,default="MSR/ae")
    parser.add_argument("--alg", type=str,default="spectral")
    parser.add_argument("--cmd", type=str,default="eval")
    parser.add_argument("--layer", type=int,default=1)
    args=parser.parse_args()
    layer_path=f"{args.nn_path}/layer_{args.layer}"
    feat_group=seq.get_group("feat")
    seqs=feat_group.read(f"{layer_path}/seqs")
    if(args.cmd=="make"):
        make_clust( seqs,
                layer_path,
                alg_type=args.alg,
                n_clusters=range(50))
    if(args.cmd=="eval"):
        layer_dir= LayerDir(layer_path)
        eval_clust( layer_dir,
                    alg_type=args.alg)