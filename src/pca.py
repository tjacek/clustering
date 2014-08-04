import numpy as np

def pca(points):
    data=np.asmatrix(points)
    data=data.T
    return pca_algorithm(data)

class PcaConverter(object):
    def __init__(self,eigen,k=2):
        self.sort(eigen[0],eigen[1])
        self.get_projection_matrix(k)

    def sort(self,eigenvalues,eigenvectors):
	indexes={}
        for i,eigen in enumerate(eigenvalues):
	    indexes[eigen]=i
	self.eigenvalues=eigenvalues
        self.eigenvalues.sort()
        self.eigenvectors=np.zeros(eigenvectors.shape)
        for i,eigen in enumerate(eigenvalues):
            j=indexes[eigen]
	    self.eigenvectors[i]=eigenvectors[j]

    def get_projection_matrix(self,k):
        major_axsis=[]
        for eigen in self.eigenvectors[0:k]:
            major_axsis.append(eigen)
        self.matrix=np.asmatrix(major_axsis)
        return self.matrix

    def projection(self,point):
        return self.matrix*point

    def list_projection(self,points):
        projected_list=[]
        for point in points:
            projected_list.append(self.projection(point))
        return projected_list

def pca_algorithm(data):
    cov_matrix=get_cov_matrix(data)
    #print(cov_matrix)
    eigenvalues=np.linalg.eig(cov_matrix)
    print(eigenvalues)
    return PcaConverter(eigenvalues)

def get_cov_matrix(data):
    u=np.apply_along_axis(avg, axis=1, arr=data)
    n= float(data.shape[1])
    for i,u_i in enumerate(u):
        for j in  range(len(data[i])):
            data[i][j]-=u_i
    print(data)
    cov_matrix= data * data.T
    cov_matrix=(1/n)*cov_matrix
    return cov_matrix

def avg(array):
    total_sum=sum(array)
    n=float(len(array))
    return total_sum/n

def pca_test():
    points=[[90.0,60.0,90.0],
            [90.0,90.0,30.0],
            [60.0,60.0,60.0],
            [60.0,60.0,90.0],
            [30.0,30.0,30.0]]
    points2=[[90.0,90.0,60.0,60.0,30.0],
             [60.0,90.0,60.0,60.0,30.0],
             [90.0,30.0,60.0,90.0,30.0]]
    points3=[[1.0,1.0, 0.0,0.1],
             [2.0,2.0,-0.1,0.0],
             [3.0,3.0, 0.0,-0.1],
             [4.0,4.0, 0.1,0.0]]
    result=pca(points3)
    #print(result.eigenvalues)
    #print(result.eigenvectors)
    #result.projection([0.1,0.1,0.1,1.0])


pca_test()
