import clusters.algs

def get_cluster_alg(alg_type):
    if(alg_type=="spectral"):
        return clusters.algs.spectral_alg
    return clusters.algs.kmeans_alg