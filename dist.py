import numpy as np
import base

def centroid_distance(dataset):
    groups=[ dataset.get_cat(i)
             for i in range(dataset.n_cats())]
    centroids=[ np.mean(group_i,axis=0)  
               for group_i in groups]
    distances=[[np.linalg.norm(centroid_i-point_j) 
                    for point_j in groups[i]]
                for i,centroid_i in enumerate(centroids)]
    return distances,centroids

exp=base.read_exp("simple_cnn")
feat=exp.get_features('dense')
distances,centroids=centroid_distance(feat)