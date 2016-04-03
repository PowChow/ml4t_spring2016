"""
Generates dataset for knn model
"""

import numpy as np
import csv
import math

def CirclePoints(x=0., y=0., r=3, n=100):
    return [(math.cos(2*math.pi/n*x)*r, math.sin(2*math.pi/n*x)*r) for x in xrange(0,n)]

def main():
    k = 3
    X = []
    for i in xrange(k):
        radius = np.random.randint(low=3, high=10, size=1)
        cp = CirclePoints(x=0., y=0., r=radius, n=1000/k)
        X.append(np.reshape(cp, newshape=(len(cp),2)))


    with open('./Data/best4knn.csv', 'w') as f:
        w = csv.writer(f, delimiter=',')
        for i in range(0, len(X)):
            for j in range(0, X[i].shape[0]):
                w.writerow([X[i][j][0], X[i][j][1], i])

if __name__== "__main__":
    main()