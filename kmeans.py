from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import matplotlib.pyplot as plt
import cnn

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

def xy_exp(clusters=None):
    if(clusters is None):
        clusters=[4,6,8,10,12]
    x,y=[],[]
    for n_clusters in clusters:
        avg_i=eval_cluster(n_clusters)
        x.append(n_clusters)
        y.append(avg_i)
    print(x)
    print(y)

def neuron_exp():
    neurons=[64,128,256,512]
    x,y=[],[]
    for neuron_i in neurons:
        avg_i=eval_cluster(n_neurons=neuron_i)
        x.append(neuron_i)
        y.append(avg_i)
    print(x)
    print(y)
    plot_xy(x,y)

def plot_xy(x,y):
    plt.plot(x,y, 'o-r')
    plt.ylabel("neurons")
    plt.ylabel('silhouette')
    plt.show()

neuron_exp()