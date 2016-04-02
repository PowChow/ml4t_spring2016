"""
kNN learner based on template by Prof Tucker Balch
related to linear regression for machine learning for trading
gaTech
"""

import numpy as np

class KNNLearner(k=3, object):

    def __init__(self, k=3, verbose=False):
        self.verbose = verbose
        self.k = k
        pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner, append to existing
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # slap on 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
        newdataX[:,0:dataX.shape[1]]=dataX
        newdataY = dataY

        # if bagging selected then add distribution of points here

        #self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        #create zero array with size of points for Y predicted values
        predY = np.zeros(shape=(points.shape[0],))
        t_dist = np.zeros(shape=(self.newdataX.shape[0],))

        # for loop to calculate Euclidean distance between query point and all
        for p in points:
            for t in self.newdataX:
                t_dist[t] = np.linalg.norm(p, i)

            sortdistindex = t_dist.argsort(axis=0)[:self.k]
            predY[p] = np.average(newdataY[sortdistindex], axis=0)

            t_dist = np.zeros(shape=(self.newdataX.shape[0],)) #reset temp distance array

        # if boosting add query here

        # return array of predicted Ys
        return predY

if __name__== "__main__":
    print "the secret clue is 'zzyzx'"
