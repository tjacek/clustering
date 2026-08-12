import clusters.algs
import clusters.score

def get_cluster_alg(alg_type):
    if(alg_type=="spectral"):
        return clusters.algs.spectral_alg
    return clusters.algs.kmeans_alg


def get_score(score_type):
    if(score_type=="silh"):
        return clusters.score.silhouette
    if(score_type=="homo"):
        return clusters.score.homogeneity
    raise Exception(f"Unknown score fun {score_type}")