import numpy as np

def default_cluster(size,dim=2):
    means=np.zeros(dim)
    variances=np.ones(dim)
    return create_cluster(size,dim,means,variances)

def create_cluster(size,dim,means,variances):
    create_point=lambda :random_point(dim,means,variances)
    cluster=[create_point() for i in range(size)]
    return cluster

def random_point(dim,means,variances):
    point=np.zeros(dim)
    for i,mean in enumerate(means):
	var=variances[i]
        point[i]=np.random.normal(mean,var)
    return point

def four_clusters(size=10,k=10.0):
    var=np.asarray([2.0,2.0])
    means1=np.asarray([ k, k])
    means2=np.asarray([ k,-k])
    means3=np.asarray([-k, k])
    means4=np.asarray([-k,-k])
    points=[]
    points+=create_cluster(size,2,means1,var)
    points+=create_cluster(size,2,means2,var)
    points+=create_cluster(size,2,means3,var)
    points+=create_cluster(size,2,means4,var)
    return points

if __name__ == "__main__":
    print(four_clusters(size=10,k=10.0))
