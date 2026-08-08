import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples,silhouette_score
from sklearn.metrics import homogeneity_score
from tqdm import tqdm
import argparse
import labels,plot,utils

class LayerDir(object):
    def __init__(self,path):
        self.path=path
        self._seqs=None

    @property
    def seqs(self):
        if(self._seqs is None):
            read=labels.FeatSeqGroup.read
            self._seqs=read(f"{self.path}/seqs")
        return self._seqs

    def labelings(self,alg_type):
        regex= rf"{alg_type}_\d+"
        return utils.find_paths(layer_path,regex )

class ClusterAsig(object):
    def __init__( self,
                preclustr,
                labels,
                dynamic):
        self.preclustr=preclustr
        self.labels=labels
        self.dynamic=dynamic

    def score(self,score_type):
        if(score_type=="homo"):
            s=homogeneity_score( self.preclustr.cats,
                                 self.labels)
        else:
            s= silhouette_score( self.preclustr.frames,
                                 self.labels,
                                 metric='euclidean')
        print(s)
        return s

    def get_labels(self,seqs):
        if(self.dynamic is None):
            return self.from_order()
        else:
            return self.from_seqs(seqs)

    def from_order(self):
        def helper(i):
            return self.labels[i]
        order=self.preclustr.order_labeling
        return order.map(helper)

    def from_seqs(self,seqs):
        return seqs.map_seq(self.dynamic,
                            group_type=labels.LabelingGroup)

class DynamicKmeans(object):
    def __init__(self,centroids):
        self.centroids=centroids
  
    def __call__(self,feat_seq):
        def helper(frame_i):
            dist=np.linalg.norm(self.centroids-frame_i,axis=1)
            return np.argmin(dist)
        return [ helper(frame_i)
                   for frame_i in feat_seq]

def get_cluster_alg(alg_type):
    if(alg_type=="spectral"):
        return spectral_alg
    return kmeans_alg

def kmeans_alg(preclustr,
               n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto")
    kmeans.fit(preclustr.frames)
    dynamic=DynamicKmeans(kmeans.cluster_centers_)
    return ClusterAsig(  preclustr=preclustr,
    	                 labels=kmeans.labels_,
                         dynamic=dynamic)

def spectral_alg(preclustr,
               n_clusters=2):
    alg = SpectralClustering(n_clusters=n_clusters, 
                             assign_labels="kmeans",
                             n_neighbors=10,
                             random_state=0)
    alg.fit(preclustr.frames)
    return ClusterAsig(  preclustr=preclustr,
                         labels=alg.labels_,
                         dynamic=None)

def make_clust( seqs,
                layer_path,
                n_clusters=None,
                alg_type="kmeans"):
    train,test=seqs.split()
#    train,test=seqs,seqs
    alg=get_cluster_alg(alg_type)
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
                alg_type="kmeans"):
    scores,sizes=[],[]
    frames=np.array(layer_dir.seqs.flatten())
    for path_i in tqdm(layer_dir.labelings(alg_type)):
        labeling_i=labels.LabelingGroup.read(path_i)
        labels_i=labeling_i.flatten()
        score_i=silhouette_score( frames,
                                  labels_i)
        scores.append(score_i)
        sizes.append(len(scores)+1)
    plot.scatter( sizes, scores, 
                  title=alg_type,
                  xlabel="n_clusters",
                  ylabel="Silhouette")    
    print(scores)

#    if(verbose):
#        plot.show_heatmap( cls_labels.tf_idf(),
#                           title=clust_name)
#    cls_labels.as_symbols("tf-idf",verbose=True)

def find_number( precluster,
	             alg_type="kmeans",
                 score_type="Silhouette",
	             max_cluster=50):
    alg=get_cluster_alg(alg_type)
    all_scores=[]
    sizes=np.array(range(max_cluster))+2
    assig=[ alg(precluster,k) 
              for k in tqdm(sizes)]
    scores=np.array([assig_i.score(score_type) 
    	        for assig_i in tqdm(assig)])
    print(scores)
    k=np.argmax(scores)
    plot.scatter( sizes, scores, 
                  title=alg_type,
                  xlabel="n_clusters",
                  ylabel=score_type)
    return sizes[k],assig[k]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn_path", type=str,default="MSR/ae")
    parser.add_argument("--alg", type=str,default="kmeans")
    parser.add_argument("--cmd", type=str,default="eval")
    parser.add_argument("--layer", type=int,default=1)
    args=parser.parse_args()
    layer_path=f"{args.nn_path}/layer_{args.layer}"
    seqs= labels.FeatSeqGroup.read(f"{layer_path}/seqs")
    if(args.cmd=="make"):
        make_clust( seqs,
                layer_path,
                alg_type=args.alg,
                n_clusters=range(50))
    if(args.cmd=="eval"):
        layer_dir= LayerDir(layer_path)
        eval_clust( layer_dir,
                    alg_type=args.alg)