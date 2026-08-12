from sklearn.metrics import silhouette_samples,silhouette_score
from sklearn.metrics import homogeneity_score

def silhouette(layer,labels):
    return silhouette_score( layer.frames,
                             labels,
                             metric='euclidean')

def homogeneity(layer,labels):
    return homogeneity_score( layer.cats,
                              labels)