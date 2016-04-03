"""
Generates dataset for knn model
"""

import numpy as np
import csv
import math

def CirclePoints(c=np.array([2.0, 5.0]), r=3, n=100):
    ps = np.array([( c[0] + math.cos(2*math.pi/n*x)*r, c[1]+math.sin(2*math.pi/n*x)*r) for x in xrange(0,n)])
    # points = np.ones(shape=(ps.shape[0], ps.shape[1]))
    #
    # for i in xrange(0, points.shape[0]):
    #     points[i,:] = ps[i] + c[:]
    # return points
    return ps

def main():
    np.random.seed(125)
    k = 3
    X = []

    for i in xrange(k):
        radius = np.random.randint(low=3, high=10, size=1)
        x1 = np.random.random_integers(0, 10)
        y1 = np.random.random_integers(-10, 10)
        center = np.array([x1, y1])
        print center
        cp = CirclePoints(c=center, r=radius, n=1000/k)
        X.append(np.reshape(cp, newshape=(len(cp),2)))


    with open('./Data/best4knn.csv', 'w') as f:
        w = csv.writer(f, delimiter=',')
        for i in range(0, len(X)):
            for j in range(0, X[i].shape[0]):
                w.writerow([X[i][j][0], X[i][j][1], i])

if __name__== "__main__":
    main()