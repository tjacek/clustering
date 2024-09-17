import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

class Torus(object):
    def __init__(self,R=5.0,r=1.0):
        self.R=R
        self.r=r

    def __call__(self,n=500):
        return [ self.single() for i in range(n)]  

    def single(self):
    	theta,psi=np.random.uniform(low=0.0, 
                                    high=2*np.pi, 
                                    size=2)
    	x=(self.R+self.r*np.cos(theta))*np.cos(psi)
    	y=(self.R+self.r*np.cos(theta))*np.sin(psi)
    	z= self.r*np.sin(theta)
    	return np.array([x,y,z])

def get_complex(points,max_edge_length=2.7):
    rips_complex = gd.RipsComplex(points=points, 
                                  max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    return simplex_tree.persistence(min_persistence=0.4)

def show_bar(points,max_edge_length=2.7):
    diag=get_complex(points,max_edge_length)
    gd.plot_persistence_barcode(diag)
    plt.show()

def show_diagram(points,max_edge_length=2.7):
    diag=get_complex(points,max_edge_length)
    ax = gd.plot_persistence_diagram(diag)
    ax.set_title("Persistence diagram of a torus")
    ax.set_aspect("equal")  # 
    plt.show()

if __name__ == '__main__':
    t=Torus()
    points=t()
    show_bar(points)