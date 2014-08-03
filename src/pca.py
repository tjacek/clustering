import numpy as np

def pca(points):
    data=np.asmatrix(points)
    return pca_algorithm(data)

def pca_algorithm(data):
    cov_matrix=get_cov_matrix(data)
    return cov_matrix

def get_cov_matrix(data):
    u=np.apply_along_axis(avg, axis=1, arr=data)
    for i,u_i in enumerate(u):
        for j in  range(len(data[i])):
            data[i][j]-=u_i
    cov_matrix= data * data.T
    return cov_matrix

def avg(array):
    total_sum=sum(array)
    n=float(len(array))
    return total_sum/n

def pca_test():
    points=[[1.0,2.0,3.0,4.0],
            [2.0,3.0,4.0,5.0],
            [3.0,4.0,5.0,6.0]]
    result=pca(points)
    print(result)

pca_test()
