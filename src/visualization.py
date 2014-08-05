import numpy as np,matplotlib.pyplot as plt

colors=['r','b','y','c','g']

def visualize_clusters(clusters):
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='2d')
    for i,cluster in enumerate(clusters):
       cords= cluster.as_cordinates()
       ax.scatter(cords[0],cords[1],c=colors[i],marker='o')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()
