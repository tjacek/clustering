import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components


import clusters.core 

def kmeans_alg(preclustr,
               n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto")
    kmeans.fit(preclustr.frames)
    dynamic=DynamicKmeans(kmeans.cluster_centers_)
    return clusters.core.ClusterAsig(  preclustr=preclustr,
    	                 labels=kmeans.labels_,
                         dynamic=dynamic)

class DynamicKmeans(object):
    def __init__(self,centroids):
        self.centroids=centroids
  
    def __call__(self,feat_seq):
        def helper(frame_i):
            dist=np.linalg.norm(self.centroids-frame_i,axis=1)
            return np.argmin(dist)
        return [ helper(frame_i)
                   for frame_i in feat_seq]

def spectral_alg(preclustr,
               n_clusters=2):
    alg = SpectralClustering(n_clusters=n_clusters,
                              affinity="nearest_neighbors", 
                             assign_labels="kmeans",
                             n_neighbors=10,
                             random_state=0)
    alg.fit(preclustr.frames)
    return clusters.core.ClusterAsig(  preclustr=preclustr,
                         labels=alg.labels_,
                         dynamic=None)

class CustomSpectral(object):
    def __call__( self,
                  preclustr,
                  n_clusters=2):
        graph = kneighbors_graph( preclustr.frames, 
                                  
                                  n_neighbors=10, 
                                  include_self=False)
        n_components, labels = connected_components(graph, directed=False)
        if(n_components>1):
            raise Exception(n_components,n_clusters)
        alg= SpectralClustering(#affinity=graph,
                                 affinity="nearest_neighbors",
                                  n_clusters=n_clusters,
                                  n_neighbors=10,
                                  random_state=0)
        alg.fit(preclustr.frames)
        raise Exception(max(alg.labels_))

