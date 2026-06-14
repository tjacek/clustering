import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import base,cnn,cluster,exp

def gauss_diff(img,sigma=5):
	return np.abs(img-gaussian_filter(img,sigma=sigma))

def gauss(img,sigma=5):
	return  gaussian_filter(img,sigma=sigma)

def freq_iter( fun,
               data,
               model,
               sigma=None):
    if(sigma is None):
        sigma=[1,2,3,4,5]
    for sigma_i in sigma:
        data_i=data(lambda img: fun(img,sigma_i))
        feat=model.extract(data_i)
        yield feat

def cluster_dist(features,metric="L2"):
    n_clusters=features.n_cats()
    clust,clust_assig=cluster.kmeans_alg(features,
                                         n_clusters=n_clusters)   
    q=clust_assig.quality(metric=metric)
    show_heat(q,title=metric)

def simple_exp(out_path):
    features=exp.get_features(out_path)
    n_clusters=features.n_cats()
    clust,clust_assig=cluster.kmeans_alg(features,
                                         n_clusters=n_clusters)  
    for clust_i in clust_assig.all_clusters():
        print(clust_i.feat.mean_norm())

def freq_exp(out_path):
    features=exp.get_features(out_path)
    cluster_dist(features,metric="L2")
    return
    purity_hist=clust.new_purity( clust_assig,
                                  feat,
                                  train)
    show_heat(purity_hist)
    n_clusters=purity_hist.shape[0]
    full_purity=[purity_hist]
    for feat_i in freq_iter(gauss,data.train,model):
        clust_assig_i=clust(feat_i,data.train)
        purity_i=clust_assig_i.purity(n_clusters)
        full_purity.append(purity_i)
    full_purity=np.array(full_purity)
    for i,purity_by_cat in enumerate(full_purity.T):
        show_heat(purity_by_cat.T,str(i))

def show_heat(X,title=None):
    sns.heatmap(X, 
                annot=True, fmt="g", cmap='viridis')
    if(title):
        plt.title(title)
    plt.show()

def pca_feats(out_path,metric="L2"):
    features=exp.get_features(out_path,0.01)
#    cluster_dist(features,metric)
    pca_features=features.to_pca()
    pca_features.plot(2,1)
#    cluster_dist(pca_features,metric)


#freq_exp("cnn_test.keras")
pca_feats("cnn_test.keras")