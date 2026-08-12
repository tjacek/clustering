import clusters.algs
import clusters.score

def get_cluster_alg(alg_type):
    if(alg_type=="spectral"):
#        return clusters.algs.CustomSpectral()
        return clusters.algs.spectral_alg
    return clusters.algs.kmeans_alg

def get_score(score_type):
    if(score_type=="silh"):
        return clusters.score.silhouette
    if(score_type=="norm_mutual"):
        return clusters.score.normalized_mutual_info
    if(score_type=="adj_mutual"):
        return clusters.score.adjusted_mutual_info
    if(score_type=="homo"):
        return clusters.score.homogeneity
    if(score_type=="db_index"):
        return clusters.score.davies_bouldin
    if(score_type=="var_ratio"):
        return clusters.score.var_ratio
    raise Exception(f"Unknown score fun {score_type}")