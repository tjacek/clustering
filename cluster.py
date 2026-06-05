import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_samples,silhouette_score
import itertools
import cv2
import cProfile
import base,utils



class Cluster(object):
    def __init__(self,X,y,feat):
        self.X=X
        self.y=y
        self.feat=feat
    
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
        for i,x_i in enumerate(self.X):
            out_ij=f"{out_path}/{i}.png"
            cv2.imwrite(out_ij,x_i)

    def mean(self):
        return np.mean(self.X,axis=0)

class ClusterAsig(object):
    def __init__( self,
                  labels,
                  data,
                  feat):
        self.labels=labels
        self.data=data
        self.feat=feat
    
    def n_clusters(self):
        return max(self.labels)+1

    def all_clusters(self):
        n=self.n_clusters()
        return [ self.get_cluster(i) 
                     for i in range(n)]

    def get_cluster(self,i):
        indexes=(self.labels==i)
        x_i=self.data.X[indexes]
        feat_i=self.feat[indexes]
        y_i=self.labels[indexes]
        return Cluster(x_i,y_i,feat_i)

    def clust_size(self,clusters):
        clusters=range(self.n_clusters())
        sizes=[]
        for clust_i in clusters:
            indexes=(self.labels==clust_i)
            sizes.append(len(indexes))
        return sizes

    def clust_hist(self,clusters=None):
        if(clusters==None):
            clusters=range(self.n_clusters())
        if(type(clusters)==int):
            clusters=[clusters]
        for clust_i in clusters:
            indexes=(self.labels==clust_i)
            y_i=self.data.y[indexes]
            plt.hist(y_i)
            plt.show()

    def quality(self,metric=None):
        metric=base.get_metric("L2",True)
        clusters=self.all_clusters()
        dist_matrix=np.zeros((len(clusters),len(clusters)))
        for i,j in utils.index_pairs(clusters):
            print(i,j)
            clus_i=clusters[i]
            clus_j=clusters[j]
            profile_metric(clus_i.feat,
                           clus_j.feat,
                           metric_type="L1")
            return
#            dist_j= all_dist(clus_i,clus_j,metric)
#            dist_matrix[i][j]=np.mean(dist_j)
        return dist_matrix

    def purity(self,n_clusters=None):
        n_cats=self.data.n_cats()
        if(n_clusters is None):
            n_clusters=self.n_clusters()
        purity_hist=np.zeros((n_clusters,n_cats))
        for clust_i,y_i in zip(self.labels,self.data.y):
            purity_hist[clust_i][y_i]+=1
        return purity_hist
    
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

def kmeans_alg(data,
               feat,
               n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto").fit(feat)
    assig=ClusterAsig( labels=kmeans.labels_,
                       data=data.train,
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
