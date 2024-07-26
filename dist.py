import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import seaborn as sn
import base

def centroid_distance(dataset):
    centroids,groups=compute_centroids(dataset)
    distances=[[np.linalg.norm(centroid_i-point_j) 
                    for point_j in groups[i]]
                for i,centroid_i in enumerate(centroids)]
    return distances,centroids

def compute_centroids(dataset):
    groups=[ dataset.get_cat(i)
             for i in range(dataset.n_cats()+1)]
    centroids=[ np.mean(group_i,axis=0)  
               for group_i in groups]
    return centroids,groups

def k_density(distances):
    kernel_alg=KernelDensity(kernel="gaussian", 
                    bandwidth=1.0)#0.5)
    X_plot = np.linspace(0, 140, 1000)[:, np.newaxis]
    all_dens=[]
    for distance_i in distances:    
        X=np.array(distance_i).reshape(-1, 1)
        kde = kernel_alg.fit(X)
        log_dens = kde.score_samples(X_plot)
        all_dens.append(log_dens)
    return all_dens,X_plot

def show_plot(all_dens,X_plot):
    fig, ax = plt.subplots()
    for dens_i in all_dens:
        ax.plot(X_plot[:, 0],
                np.exp(dens_i),
#    linestyle="-",
#    label="0"
        )
    plt.show()

def centroid_pairs(dataset):
    centroids,groups=compute_centroids(dataset)
    distances=[[np.linalg.norm(centroid_i-centroid_j) 
                for centroid_j in centroids]
                    for centroid_i in centroids]
    distances=np.array(distances)
    print(distances)
    sn.heatmap(distances, annot=True)
    plt.show()
    dis_min=np.argmin(distances,axis=0)
    dis_max=np.argmax(distances,axis=0)
    for i,min_i in enumerate(dis_min):
        max_i=dis_max[i]
        print(f"{i}:{min_i}:{max_i}")

feat=base.read_dataset("cnn_feats.npz")
centroid_pairs(feat)

#exp=base.read_exp("simple_cnn")
#feat=exp.get_features('dense')
#feat=base.read_dataset("cnn_feats.npz")
#distances,centroids=centroid_distance(feat)
#all_dens,X_plot=k_density(distances)
#show_plot(all_dens,X_plot)