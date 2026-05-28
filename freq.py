import base
from scipy.ndimage import gaussian_filter
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
    print(clust_assig.centroids)

freq_exp("cnn_test.keras")