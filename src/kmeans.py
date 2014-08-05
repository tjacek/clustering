import math,numpy as np

def kmeans(points,k=2,dim=2):
    clusters=[Cluster(dim) for i in range(k)]
    return clusters

class Cluster(object):
    def __init__(self,dim=2):
        self.dim=dim
        self.points=[] #Punkty maja typ array
        self.means=np.zeros(dim)

    def means(self):
        for i in range(self.dim):
	    get_cord=lambda x:x[i]
            dim_i=map(get_cord,self.points)
            self.means[i]=sum(dim_i)/self.size()

    def size():
        return float(len(self.points))

def euclidean_metric(x,y):
    value=0.0
    for x_i,y_i in zip(x,y):
        value+=(x_i-y_i)**2
    return math.sqrt(value)

if __name__ == "__main__":
    print(euclidean_metric([1.0,1.0,1.0],[2.0,2.0,2.0]))
    kmeans([])
