import sklearn.metrics as sk_metrics

def silhouette(layer,labels):
    return sk_metrics.silhouette_score( layer.frames,
                                        labels,
                                        metric='euclidean')

def davies_bouldin(layer,labels):
    return sk_metrics.davies_bouldin_score(layer.frames, 
                                           labels)

def var_ratio(layer,labels):
    return sk_metrics.calinski_harabasz_score( layer.frames, 
                                               labels)

def homogeneity(layer,labels):
    return sk_metrics.homogeneity_score( layer.cats,
                                         labels)

def normalized_mutual_info(layer,labels):
    return sk_metrics.normalized_mutual_info_score( layer.cats,
                                                    labels)

def adjusted_mutual_info(layer,labels):
    return sk_metrics.adjusted_mutual_info_score( layer.cats,
                                                  labels)