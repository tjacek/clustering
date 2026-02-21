from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score
import cnn

def cluster(n_clusters=3):
    model,data=cnn.simple_exp()
    feat=model.extract(data.train)
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto").fit(feat)
    return kmeans.labels_,data,feat


clust,data,feat=cluster(n_clusters=3)
silhouette_avg = silhouette_score(feat, clust)
sample_silhouette_values = silhouette_samples(feat, 
	                                          clust)
print(silhouette_avg)