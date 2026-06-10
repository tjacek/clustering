import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import base,cnn,cluster

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

def simple_exp():
    data=base.get_minst_dataset()
    s_data=data.train.subsample(0.03)
    diff_data=s_data(gauss)
    diff_data.save("gauss")
 
def freq_exp(out_path):
    data=base.get_minst_dataset()
    model=cnn.ConvNN.get_model(out_path,data)
    feat=model.extract(data.train)
    features=base.Features(feat,data.train)
    n_clusters=features.n_cats()
    clust,clust_assig=cluster.kmeans_alg(features,
                                         n_clusters=n_clusters)   
    q=clust_assig.quality(metric="cos")
    show_heat(q)
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

def pca_feats(out_path):
    data=base.get_minst_dataset()
    model=cnn.ConvNN.get_model(out_path,data)
    feat=model.extract(data.train)
    features=base.Features(feat,data.train)
    pca=features.to_pca()
    print(pca.cum_var())
#    print(pca.info/np.sum(pca.info))
#    print(np.cumsum(pca.info/np.sum(pca.info)))

freq_exp("cnn_test.keras")
#pca_feats("cnn_test.keras")