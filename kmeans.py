import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.pyplot as plt
import cnn

class ExpResults(object):
    def __init__( self,
                  clusters,
                  neurons,
                  metric):
        self.dict={"clusters":clusters,
                   "neurons":neurons,
                   "metric":metric}

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

class ClusterAsig(object):
    def __init__( self,
                  labels,
                  data,
                  feat):
        self.labels=labels
        self.data=data
        self.feat=feat

    def get_cluster(self,i):
        indexes=(self.labels==i)
        return data.X[indexes]


def cluster(n_clusters=3,
            n_neurons=512):
    model,data=cnn.simple_exp(n_neurons=n_neurons)
    feat=model.extract(data.train)
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto").fit(feat)
    return ClusterAsig(labels=kmeans.labels_,
                       data=data,
                       feat=feat)


def eval_cluster(n_clusters=3,
                 n_neurons=512):
    clustg=cluster(n_clusters,n_neurons)
    silhouette_avg = silhouette_score(feat, clust.labels)
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

#def plot_xy(x,y):
#    plt.plot(x,y, 'o-r')
#    plt.ylabel("neurons")
#    plt.ylabel('silhouette')
#    plt.show()

exp=xy_exp("neuron")
exp.plot()