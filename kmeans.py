from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import cnn

def cluster(n_clusters=3):
    cnn
    model,data=cnn.simple_exp()
    feat=model.extract(data.train)
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto").fit(feat)
    return kmeans.labels_,data,feat


def eval_cluster(n_clusters=3):
    clust,data,feat=cluster(n_clusters)
    silhouette_avg = silhouette_score(feat, clust)
    sample_silhouette_values = silhouette_samples(feat, 
	                                          clust)
    return silhouette_avg

def xy_exp():
    values=[4,6,8,10,12]
    x,y=[],[]
    for n_clusters in values:
        avg_i=eval_cluster(n_clusters)
        x.append(n_clusters)
        y.append(avg_i)
    print(x)
    print(y)

xy_exp()