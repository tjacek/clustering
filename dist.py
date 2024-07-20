import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
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

def k_density(distances):
    kernel_alg=KernelDensity(kernel="gaussian", 
                    bandwidth=0.5)
    X_plot = np.linspace(0, 200, 1000)[:, np.newaxis]
    all_dens=[]
    for distance_i in distances:    
        X=np.array(distance_i).reshape(-1, 1)
        kde = kernel_alg.fit(X)
        log_dens = kde.score_samples(X_plot)
        all_dens.append(log_dens)
    return all_dens

exp=base.read_exp("simple_cnn")
feat=exp.get_features('dense_1')
distances,centroids=centroid_distance(feat)
k_density(distances)

#fig, ax = plt.subplots()
#kde = KernelDensity(kernel="gaussian", 
#                    bandwidth=0.5).fit(X)
#log_dens = kde.score_samples(X_plot)
#ax.plot(
#    X_plot[:, 0],
#    np.exp(log_dens),
#    linestyle="-",
#    label="0"
#)
#plt.show()