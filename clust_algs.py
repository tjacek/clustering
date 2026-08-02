import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples,silhouette_score
from tqdm import tqdm
import labels
#class C

def get_cluster_alg(alg_type):
    if(alg_type=="spectral"):
        return spectral_alg
    return kmeans_alg

def kmeans_alg(frames,
               n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto").fit(frames)
    return kmeans.labels_ 
#    assig=ClusterAsig( labels=kmeans.labels_,
#                       feat=feat)
#    clust=KMeansClust(centroids=kmeans.cluster_centers_)
#    return clust,assig

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
    for k in tqdm(range(max_cluster)):
        assig_i=alg(precluster.frames,k+2)
        score_i=silhouette_score(precluster.frames,
	                           assig_i,
	                           metric='euclidean')
        all_scores.append(score_i)
    all_scores=np.array(all_scores)
    print(all_scores)
    x=np.array(range(max_cluster))+2
    plt.scatter(x, all_scores,  alpha=0.5)
    plt.show()

seqs= labels.FeatSeqGroup.read("MSR/seq")
precluster=seqs.as_precluster()
find_number(precluster)