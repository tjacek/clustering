import numpy as np
from tqdm import tqdm
import argparse
import clusters
import seq
import plot
import utils

class LayerDir(utils.DirProxy):
    def __init__(self,path):
        super().__init__(path)
        self.frame_info=None

    def labelings(self,alg_type):
        return utils.top_files(self[alg_type])
#        return utils.find_paths(self.dir_path,regex)

    @classmethod
    def make(cls,nn_path,layer):
        return cls(f"{nn_path}/layer_{layer}")
    
    @property
    def info(self):
        if(self.frame_info is None):
            feat_group=seq.get_group("feat")
            seqs=feat_group.read(self["seqs"])
            self.frame_info=seqs.info()
        return self.frame_info

    def clust( self,
               alg_type,
               n_clusters):
        clust_path=self[alg_type]
        for k in n_clusters:
            if(k<8):
                continue
            path_k=f"{clust_path}/{k}"
            yield k,path_k

def make_clust( layer_dir,
                n_clusters=None,
                alg_type="spectral"):
    seqs=layer_dir.seqs()
    if(alg_type=="spectral"):   
        train,test=seqs,seqs
    else:
         train,test=seqs.split()
    alg=clusters.get_cluster_alg(alg_type)
    precluster=train.info()
    if( type(n_clusters)==int):
        n_clusters=[n_clusters]
    cls_iter=layer_dir.clust( alg_type,
                              n_clusters)
    for k,path_k in tqdm(cls_iter):
        assig=alg(precluster,k)
        cls_labels=assig.get_labels(seqs)
        cls_labels.save(path_k)

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
    parser.add_argument("--nn_path", type=str,default="MSR/sim")
    parser.add_argument("--alg", type=str,default="spectral")
    parser.add_argument("--cmd", type=str,default="eval")
    parser.add_argument("--layer", type=int,default=1)
    args=parser.parse_args()
    layer_dir= LayerDir.make(args.nn_path,args.layer)
    if(args.cmd=="make"):
        make_clust( layer_dir,
                    alg_type=args.alg,
                    n_clusters=range(50))
    if(args.cmd=="eval"):
        eval_clust( layer_dir,
                    alg_type=args.alg)