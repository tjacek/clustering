import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples,silhouette_score
from tqdm import tqdm
import labels

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

import matplotlib.pyplot as plt

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
    print(np.argmax(scores))
    plt.scatter(sizes,scores,alpha=0.5)
    plt.xlabel("n_clusters")
    plt.ylabel("Silhouette")
    plt.title(alg_type)
    plt.grid(alpha=0.7)
    plt.show()


seqs= labels.FeatSeqGroup.read("MSR/seq")
precluster=seqs.as_precluster()
find_number(precluster)