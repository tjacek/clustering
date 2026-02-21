from sklearn.cluster import KMeans
import cnn

def cluster(n_clusters=3):
    model,data=cnn.simple_exp()
    feat=model.extract(data.train)
    kmeans = KMeans(n_clusters=n_clusters, 
	                random_state=0, 
	                n_init="auto").fit(feat)
    return kmeans.labels_,data.train.y


clust,y=cluster(n_clusters=3)
#(clust)
#print(y)