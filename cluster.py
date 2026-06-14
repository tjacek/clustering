import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples,silhouette_score
import itertools
import cv2
import cProfile
import base,utils

class Cluster(object):
    def __init__(self,feat):
        self.feat=feat
    
    def __len__(self):
        return self.feat.shape[0]

    def centroid(self):
        return np.mean(self.feat,axis=0)
    
    def central(self):
        center=self.centroid()
        dist=[ np.linalg.norm(center-feat_i,ord=2)
                 for feat_i in self.feat]
        k=np.argmax(dist)
        return self.feat[k]
    
    def save(self,out_path):
        utils.make_dir(out_path)
        for i,x_i in enumerate(self("X")):
            out_ij=f"{out_path}/{i}.png"
            cv2.imwrite(out_ij,x_i)

    def mean(self):
        return np.mean(self("X"),axis=0)
    
    def dist(self,feat,metric):  
        dist=0
        for x_i in feat:
            dist_i=metric(self.feat,x_i)
            dist+=np.mean(dist_i)
#        size=len(self)*feat.shape[0]
        return dist/len(self)

class ClusterAsig(object):
    def __init__( self,
                  labels,
                  feat):
        self.labels=labels
        self.feat=feat
    
    def n_clusters(self):
        return max(self.labels)+1

    def all_clusters(self):
        n=self.n_clusters()
        return [ self.get_cluster(i) 
                     for i in range(n)]
    def get_cluster(self,i):
        indices=(self.labels==i)
        new_feat=self.feat.select(indices)
        return Cluster(new_feat)
    
    def clust_size(self,clusters):
        clusters=range(self.n_clusters())
        sizes=[]
        for clust_i in clusters:
            indexes=(self.labels==clust_i)
            sizes.append(len(indexes))
        return sizes
    
    def quality(self,metric="L2"):
        metric=base.get_metric(metric,True)
        clusters=self.all_clusters()
        dist_matrix=np.zeros((len(clusters),len(clusters)))
        for i,clust_i in enumerate(clusters):
            print(i)
            dist_matrix[i][i]= clust_i.dist(clust_i.feat,metric)
        for i,j in utils.index_pairs(clusters):
            print(i,j)
            clus_i=clusters[i]
            clus_j=clusters[j]
            dist_ij= clus_i.dist(clus_j.feat,metric)
            dist_matrix[i][j]=dist_ij
            dist_matrix[j][i]=dist_ij
        return dist_matrix
    
    def save(self,out_path):
        utils.make_dir(out_path)
        clusters=self.all_clusters()
        for i,cluster_i in enumerate(clusters):
            cluster_i.save(f"{out_path}/{i}")

    def mean_img(self,out_path):
        utils.make_dir(out_path)
        clusters=self.all_clusters()
        for i,cluster_i in enumerate(clusters):
            path_i=f"{out_path}/{i}.png"
            cv2.imwrite(path_i,cluster_i.mean())
    
    def purity(self):
        n_cats=self.feat.n_cats()
        n_clusters=self.n_clusters()
        purity_hist=np.zeros((n_clusters,n_cats))
        for clust_i,y_i in zip(self.labels,
                               self.feat("y")):
            purity_hist[clust_i][y_i]+=1
        return purity_hist

class KMeansClust(object):
    def __init__(self,centroids):
        self.centroids=centroids
    
    def __call__(self,feat,data=None):
        labels=[ self.assig_new(feat_i)
                 for feat_i in feat]
        return ClusterAsig( labels=labels,
                            data=data,
                            feat=feat)
    
    def assig_new(self,x):
        dist=[np.linalg.norm(x-c_i, ord=2) 
                   for c_i in self.centroids ]
        return np.argmin(dist)
    
    def new_purity( self,
                    clust_assig,
                    feat,
                    data):
        n_clusters=clust_assig.n_clusters()
        purity_hist= clust_assig.purity(n_clusters)
        cls2cat=np.argmax(purity_hist,axis=1)
        self.reorder(cls2cat)
        purity_hist=self(feat,data).purity(n_clusters)  
        return purity_hist

    def reorder(self,cls2cat):
        new_ord=np.argsort(cls2cat)
        print(new_ord)
        new_centroids=[self.centroids[i]  
                        for i in new_ord ]
        self.centroids=new_centroids

def kmeans_alg(feat,
               n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto").fit(feat)
    assig=ClusterAsig( labels=kmeans.labels_,
                       feat=feat)
    clust=KMeansClust(centroids=kmeans.cluster_centers_)
    return clust,assig

def spectral_alg(data,
               feat,
               n_clusters=2):
    clust = SpectralClustering(n_clusters=n_clusters, 
                    assign_labels='discretize',
                    random_state=0).fit(feat)
    return ClusterAsig(labels=kmeans.labels_,
                       data=data.train,
                       feat=feat)

def all_dist(x,y,metric):
    distances=[]
    for i,j in utils.pair_iter(x,y): 
        x_i,y_j=x.feat[i],y
        distances.append(metric(x_i,y_i))
    return distances

def profile_metric(feat_i,feat_j,metric_type="L2"):
    metric=base.get_metric(metric_type,True)
    def new_fun():
        distance=0
        for x_i in feat_i:
            distance+=metric(feat_j,x_i)
        return distance
    cProfile.runctx('new_fun()', globals(), locals())
    metric=base.get_metric(metric_type,False)
    def old_fun():
        distance=0
        for x_i in feat_i:
            for y_i in feat_j:
                distance=metric(x_i,y_i)
        return distance
    cProfile.runctx('old_fun()', globals(), locals())