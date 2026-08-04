import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples,silhouette_score
from tqdm import tqdm
import argparse
import labels,plot,utils

class ClusterAsig(object):
	def __init__( self,
		          preclustr,
		          labels):
		self.preclustr=preclustr
		self.labels=labels

	def score(self):
		return silhouette_score( self.preclustr.frames,
	                             self.labels,
	                             metric='euclidean')
	def cls_labeling(self):
		def helper(i):
			return self.labels[i]
		order=self.preclustr.order_labeling
		return order.map(helper)

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
    return ClusterAsig(  preclustr=preclustr,
    	                 labels=kmeans.labels_)

def spectral_alg(data,
               feat,
               n_clusters=2):
    clust = SpectralClustering(n_clusters=n_clusters, 
                    assign_labels='discretize',
                    random_state=0).fit(feat)
    return ClusterAsig(labels=kmeans.labels_,
                       data=data.train,
                       feat=feat)

def make_clusteing( seqs,
                    layer_path,
                    n_clusters=None,
                    alg_type="kmeans",
                    verbose=True):
    precluster=seqs.as_precluster()
    if(n_clusters is None):
        n_clusters,assig=find_number(precluster)
    else:
        alg=get_cluster_alg(alg_type)
        assig=alg(precluster,n_clusters)
    cls_labels=assig.cls_labeling()
    clust_name=f"{alg_type}_{n_clusters}"
    cls_labels.save(f"{layer_path}/{clust_name}")
    if(verbose):
        plot.show_heatmap( cls_labels.tf_idf(),
                           title=clust_name)
#    cls_labels.as_symbols("tf-idf",verbose=True)

def find_number( precluster,
	             alg_type="kmeans",
	             max_cluster=50):
    alg=get_cluster_alg(alg_type)
    all_scores=[]
    sizes=np.array(range(max_cluster))+2
    assig=[ alg(precluster,k) 
              for k in tqdm(sizes)]
    scores=np.array([assig_i.score() 
    	        for assig_i in tqdm(assig)])
    print(scores)
    k=np.argmax(scores)
    plot.scatter( sizes, scores, 
                  title=alg_type,
                  xlabel="n_clusters",
                  ylabel="Silhouette")
    return sizes[k],assig[k]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nn_path", type=str,default="MSR/cnn")
    parser.add_argument("--layer", type=int,default=1)
    args=parser.parse_args()
    layer_path=f"{args.nn_path}/layer_{args.layer}"
    seqs= labels.FeatSeqGroup.read(f"{layer_path}/seq")
    make_clusteing( seqs,
                    layer_path,
                    n_clusters=20)