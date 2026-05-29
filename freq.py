import base
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import base,cnn,cluster

def gauss_diff(img,sigma=5):
	return (img-gaussian_filter(img,sigma=sigma))

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
    train=data.train
    clust,clust_assig=cluster.kmeans_alg(data,
                                   feat,
                                   n_clusters=train.n_cats())
    purtity_hist= clust_assig.purity()
    for feat_i in freq_iter(gauss,data.train,model):
        print(feat_i.shape)
    
#    sns.heatmap(purtity_hist, annot=True, fmt="g", cmap='viridis')
#    plt.show()

#    print(purtity_hist)

freq_exp("cnn_test.keras")