import numpy as np

class Torus(object):
    def __init__(self,R=5.0,r=1.0):
        self.R=R
        self.r=r

    def __call__(self):
    	theta,psi=np.random.uniform(low=0.0, high=2*np.pi, size=2)
    	x=(self.R+self.r*np.cos(theta))*np.cos(psi)
    	y=(self.R+self.r*np.cos(theta))*np.sin(psi)
    	z= self.r*np.sin(theta)
    	return np.array([x,y,z])

t=Torus()
print(t())