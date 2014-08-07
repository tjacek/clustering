import numpy as np,matplotlib.pyplot as plt
import kmeans

colors=['r','b','y','c','g']

def visualize_points(points):
    series=[kmeans.as_cordinates(points)]
    visualize2D(series)

def visualize_clusters(clusters):
    to_cord=lambda cls:cls.as_cordinates()
    series=map(to_cord,clusters)
    visualize2D(series)

def visualize2D(series):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i,data in enumerate(series):
       print("OK")
       #print(list(data[0]))
       #print(list(data[1]))
       ax.scatter(data[0],data[1],c=colors[i],marker='o')
       tmp1=[1.0,2.0,3.0,4.0 ,5.0]
       tmp2=[1.0,4.0,9.0,16.0,25.0]
       #ax.scatter(tmp1,tmp2,c=colors[i],marker='o')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()
