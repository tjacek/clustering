import numpy as np

class Cluster(object):
    def __init__(self,dim=2):
        self.dim=dim
        self.points=[]
        self.means=numpy.zeros(dim)

    def means(self):
        for i in range(self.dim):
	    get_cord=lambda x:x[i]
            dim_i=map(get_cord,self.points)
            self.means[i]=sum(dim_i)/self.size()

    def size():
        return float(len(self.points))
