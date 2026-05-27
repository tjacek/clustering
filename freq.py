import base
from scipy.ndimage import gaussian_filter
import cnn

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
    cnn.ConvNN.get_model(out_path)

freq_exp("cnn_test.keras")