"""
kNN learner based on template by Prof Tucker Balch
related to linear regression for machine learning for trading
gaTech
"""

import numpy as np
import math

class KNNLearner(object):

    def __init__(self, k=3, verbose=False):
        self.verbose = verbose
        self.k = k
        self.X = []
        self.Y = []
        pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner, append to existing
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.X.extend(dataX)
        self.Y.extend(dataY)

        # if bagging selected then add distribution of points here

        #self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)

        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        arrayX = np.array(self.X)
        arrayY = np.array(self.Y)

        #create zero array with size of points for Y predicted values
        predY = []
        t_dist = np.zeros(shape=(arrayX.shape[0],))

        # for loop to calculate Euclidean distance between query point and all
        for p in points:
            i = 0
            for x in arrayX:
                t_dist[i] = np.linalg.norm(p - x) #numpy calculate distance
                #d = pow((p - x), 2)
                #t_dist[i] = math.sqrt(np.sum(d))
                i= i+1

            sortdistindex = t_dist.argsort(axis=0)[:self.k]
            predY.append(np.average(t_dist[sortdistindex]))

            t_dist = np.zeros(shape=(arrayX.shape[0],))

        # if boosting add query here

        # return array of predicted Ys
        #print predY
        return predY

if __name__== "__main__":
    print "the secret clue is 'zzyzx'"
