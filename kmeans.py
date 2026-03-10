import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.pyplot as plt
import cnn

class ExpResults(object):
    NAMES=["metric","cluster","neuron"]
    COLS={"metric":0,"cluster":1,"neuron":2}
    def __init__(self):
        self.points=[]

    def add( self,
             metric,
             cluster,
             neurons):
        point=[metric,cluster,neurons]
        self.points.append(point)

    def plot(self,x,y):
        points=np.array(self.points)
        x,y=points[x],points[y]
        plt.plot(x,y, 'o-r')
        plt.ylabel("neurons")
        plt.ylabel('silhouette')
        plt.show()

def cluster(n_clusters=3,
            n_neurons=512):
    model,data=cnn.simple_exp(n_neurons=n_neurons)
    feat=model.extract(data.train)
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto").fit(feat)
    return kmeans.labels_,data,feat


def eval_cluster(n_clusters=3,
                 n_neurons=512):
    clust,data,feat=cluster(n_clusters,n_neurons)
    silhouette_avg = silhouette_score(feat, clust)
    sample_silhouette_values = silhouette_samples(feat, 
	                                          clust)
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
#    if(clusters is None):
#        clusters=[4,6,8,10,12]
    if(param_iter=="neuron"):
        param_iter=neuron_iter()
    if(param_iter=="cluster"):
        param_iter=clust_iter()
    x,y=[],[]
    for cluster,n_neurons in param_iter:
        avg_i=eval_cluster(cluster,n_neurons)
        x.append(cluster)
        y.append(avg_i)
    print(x)
    print(y)

#def neuron_exp():
#    neurons=[64,128,256,512]
#    x,y=[],[]
#    for neuron_i in neurons:
#        avg_i=eval_cluster(n_neurons=neuron_i)
#        x.append(neuron_i)
#        y.append(avg_i)
#    print(x)
#    print(y)
#    plot_xy(x,y)

def plot_xy(x,y):
    plt.plot(x,y, 'o-r')
    plt.ylabel("neurons")
    plt.ylabel('silhouette')
    plt.show()

xy_exp("neuron")