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
    	theta,psi=np.random.uniform(low=0.0, high=2*np.pi, size=2)
    	x=(self.R+self.r*np.cos(theta))*np.cos(psi)
    	y=(self.R+self.r*np.cos(theta))*np.sin(psi)
    	z= self.r*np.sin(theta)
    	return np.array([x,y,z])

def show_bar(points):
    rips_complex = gd.RipsComplex(points=points, max_edge_length=2.7)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    diag = simplex_tree.persistence(min_persistence=0.4)
    gd.plot_persistence_barcode(diag)
    plt.show()

t=Torus()
points=t()
show_bar(points)

#ac = gd.AlphaComplex(points=points)
#st = ac.create_simplex_tree()
#bar_code= st.persistent_betti_numbers(1)
#gd.plot_persistence_diagram(bar_code);
#print(dir(st))