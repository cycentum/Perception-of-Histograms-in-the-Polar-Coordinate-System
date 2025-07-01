import numpy as np


def roundedArange(lower, upper, step, decimal):
		'''
		lower: inclusive
		upper: inclusive
		decimal: For rounding
		'''
		v=np.round(np.arange(lower, upper+step/2, step), decimal)
		v[v==0]=0 #-0.0 -> 0.0
		return v


def triangularArange(arange):
	'''
	return: [lower, lower+1, ..., upper, upper-1, ..., lower+1]
	'''
	
	triangular=np.concatenate((arange, arange[-2:0:-1]))
	return triangular


def assertSymm(x, sign, axis=None):
	'''
	x: np.array
	axis: int
	'''
	if axis is None:
		assert (x==sign*np.flip(x)).all()
	else:
		assert (x==sign*np.flip(x, axis=axis)).all()


def getAssertUnique(x):
	'''
	x: np.array
	'''
	fl=x.flatten()
	assert len(fl)>1
	assert (x==fl[0]).all()
	return fl[0]


def getAssertSingle(x):
	'''
	x: np.array
	'''
	fl=x.flatten()
	assert len(fl)==1
	return fl[0]