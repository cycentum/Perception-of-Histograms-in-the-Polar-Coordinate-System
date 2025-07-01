import numpy as np
import numpy.linalg

def pol_to_cart(angleRad, r):
	x=r*np.cos(angleRad)
	y=r*np.sin(angleRad)
	return x,y


def cart_to_pol(x, y):
	xy=np.stack((x,y),axis=1)
	r=np.linalg.norm(xy,axis=1)
	angleRad=np.arctan2(y,x)
	return angleRad,r
	

PI2=np.pi*2