"""
Generates dataset for linear regression
"""

import numpy as np
import math
import time

def main():
    m, sd = 0.0, 0.2
    X_val = np.random.normal(m, sd, size=(1000,2))
    Y = []
    for x in X_val:
        Y.append(X_val[:, 0] + np.random.rand(1))
    #Y_val = np.reshape(Y, newshape=(len(Y)-1,1))


    print X_val
    print Y


#Define the h(x) value of a point for a given w
#return +1 or -1
def evaluateg(ex,w):
	r = sum(ex_i*w_i for ex_i,w_i in zip(ex,w))
	if r > 0:
		return 1
	elif r < 0:
		return -1

if __name__== "__main__":
    main()