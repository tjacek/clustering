import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.pyplot as plt
import cv2
import cnn,utils

class ExpResults(object):
    def __init__( self,
                  clusters,
                  neurons=None,
                  metric=None):
        if(type(clusters)==dict):
            out_dict=clusters
        else:
            out_dict={"clusters":clusters,
                   "neurons":neurons,
                   "metric":metric}
        self.dict=out_dict

    def add( self,
             cluster,
             neurons,
             metric):
        self.dict["clusters"].append(cluster)
        self.dict["neurons"].append(neurons)
        self.dict["metric"].append(metric)

    def plot( self,
              x_name="neurons",
              y_name="metric"):
        x,y=self.dict[x_name],self.dict[y_name]
        plt.plot(x,y, 'o-r')
        plt.xlabel(x_name)
        plt.xlabel(y_name)
        plt.show()
    
    def plot3D( self):
        x=self.dict["clusters"]
        y=self.dict["neurons"]
        z=self.dict["metric"]
        plt.plot3D(x, y, z, 'green')
        plt.xlabel(x_name)
        plt.xlabel(y_name)
        plt.show()

    def ord_list(self):
        keys=list(self.dict.keys())
        keys.sort()
        values=[self.dict[key_i] for key_i in keys]
        return values

    def save(self,out_path):
        values=self.ord_list()
        n_lines=len(values[0])
        with open(out_path, 'w') as file:
            file.write(",".join(keys))
            for i in range(n_lines):
                raw_i=[str(value_j[i]) 
                        for value_j in values]
                file.write(",".join(raw_i))

   
    @classmethod
    def read(cls, in_path):
        with open(in_path, 'r') as file:
            lines = file.readlines()
        
            keys = lines[0].strip().split(",")
            dict = {key: [] for key in keys}
            for line in lines[1:]:
                values = line.strip().split(",")
                for key, value in zip(keys, values):
                     dict[key].append(value)
            return cls(dict)

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

    def get_cluster(self,i):
        indexes=(self.labels==i)
        return self.data.X[indexes]

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
            print(np.bincount(y_i))
            plt.hist(y_i)
            plt.show()

    def quality(self):
        return silhouette_score( self.feat, 
                                 self.labels)

    def save(self,out_path):
        utils.make_dir(out_path)
        n_clusters=self.n_clusters()
        for clust_i in range(n_clusters):
            out_i=f"{out_path}/{clust_i}"
            utils.make_dir(out_i)
            x_i=self.get_cluster(clust_i)
            for j,x_j in enumerate(x_i):
                out_ij=f"{out_i}/{j}.png"
                cv2.imwrite(out_ij,x_j)

def cluster(n_clusters=3,
            n_neurons=512):
    model,data=cnn.simple_exp(n_neurons=n_neurons)
    feat=model.extract(data.train)
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto").fit(feat)
    return ClusterAsig(labels=kmeans.labels_,
                       data=data.train,
                       feat=feat)


def eval_cluster(n_clusters=3,
                 n_neurons=512):
    clust=cluster(n_clusters,n_neurons)
    print(f"Clusters:{clust.n_clusters()}")
    silhouette_avg = silhouette_score( clust.feat, 
                                       clust.labels)
    sample_silhouette_values = silhouette_samples(clust.feat, 
	                                              clust.labels)
    return silhouette_avg

def clust_iter(cluster=None,n_neurons=512):
    if(clusters is None):
        clusters=[4,6,8,10,12]
    for cluster_i in clusters:
        yield cluster_i,n_neurons

def neuron_iter(cluster=3,n_neurons=None):
    if(n_neurons is None):
        n_neurons=[4,6,8,10,12]
    for neuron_i in n_neurons:
        yield cluster,neuron_i

def xy_exp(param_iter):
    if(param_iter=="neuron"):
        param_iter=neuron_iter()
    if(param_iter=="cluster"):
        param_iter=clust_iter()
    clusters,neurons,metric=[],[],[]
    for cluster_i,neurons_i in param_iter:
        metric_i=eval_cluster( cluster_i,
                               neurons_i)
        clusters.append(cluster_i)
        neurons.append(neurons_i)
        metric.append(metric_i)
    return ExpResults( clusters,
                       neurons,
                       metric)

def save_cluster(out_path,
                 n_clusters=3,
                 n_neurons=512):
    clust=cluster(n_clusters,n_neurons)
#    clust.save(out_path)
    clust.clust_hist()

exp=xy_exp("neuron")
exp.plot()
#save_cluster("out")