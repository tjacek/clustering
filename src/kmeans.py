import math,random,numpy as np
import cluster_generator,visualization as plot

def kmeans(points,k=2,dim=2,maxIterations=1000):
    clusters=[Cluster(dim) for i in range(k)]
    assig=init_assigment(points,clusters)
    assig_changed=True
    iterations=0
    while(assig_changed and iterations<maxIterations):
    	assig_changed=new_assigment(points,clusters,assig)
        iterations+=1
    return clusters

class Cluster(object):
    def __init__(self,dim=2):
        self.dim=dim
        self.points=[] #Punkty maja typ array
        self.means=np.zeros(dim)

    def size(self):
        return float(len(self.points))

    def add_point(self,point):
        self.points.append(point)

    def reset(self):
        self.points=[]

    def as_cordinates(self):
        return as_cordinates(self.points)   

    def compute_mean(self):
        for i in range(self.dim):
	    get_cord=lambda x:x[i]
            dim_i=map(get_cord,self.points)
            if(self.size()==0.0):
               break
            self.means[i]=sum(dim_i)/self.size()
        return self.means

def as_cordinates(points):
    n=len(points[0])
    return [get_cord(i,points) for i in range(n)]    

def get_cord(i,points):
    extract=lambda x:x[i]
    cord=map(extract,points)
    return cord

def init_assigment(points,clusters):
    k=len(clusters)
    assigment=np.zeros(len(points))
    for i,point in enumerate(points):
        cls=random.randrange(k)
        clusters[cls].add_point(point)
        assigment[i]=cls
    return assigment

def new_assigment(points,clusters,old_assigment):
    assigment_changed=False
    compute_means(clusters)
    for i,point in enumerate(points):
        cls_index=next_cluster(point,clusters)
        if(old_assigment[i]!=cls_index):
            assigment_changed=True
            old_assigment[i]=cls_index
            clusters[cls_index].add_point(point)
    return assigment_changed

def compute_means(clusters):
    for cluster in clusters:
        cluster.compute_mean()
        cluster.reset()
    return clusters

def next_cluster(point,clusters):
    distance=lambda cls:euclidean_metric(point,cls.means)
    distances = [distance(cls) for cls in clusters]
    return distances.index(min(distances))

def euclidean_metric(x,y):
    value=0.0
    for x_i,y_i in zip(x,y):
        value+=(x_i-y_i)**2
    return math.sqrt(value)

if __name__ == "__main__":
    points=cluster_generator.four_clusters()
    plot.visualize_points(points)
    #clusters=kmeans(points)
    #plot.visualize_clusters(clusters)
