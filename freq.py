import base
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
import base,cnn,cluster

def gauss_diff(img):
	return img-gaussian_filter(img, sigma=5)

def gauss(img):
	return gaussian_filter(img, sigma=5)

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
    clust_assig=cluster.kmeans_alg(data,
                                   feat,
                                    n_clusters=train.n_cats())
    purtity_hist= clust_assig.purity()
    
    sns.heatmap(purtity_hist, annot=True, fmt="g", cmap='viridis')
    plt.show()

#    print(purtity_hist)

freq_exp("cnn_test.keras")