"""
kNN learner based on template by Prof Tucker Balch
related to linear regression for machine learning for trading
gaTech
"""

import numpy as np

class KNNLearner(k=3, object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        pass # move along, these aren't the drones you're looking for

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # slap on 1s column so linear regression finds a constant term
        newdataX = np.ones([dataX.shape[0],dataX.shape[1]+1])
        newdataX[:,0:dataX.shape[1]]=dataX

        # add bagging here

        # add boosting here


        # knn save the point here later for querying
        #self.model_coefs, residuals, rank, s = np.linalg.lstsq(newdataX, dataY)
        
    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved modlel.
        """
        # for loop to calculate Euclidean distance between query point and all
        # train points

        return (self.model_coefs[:-1] * points).sum(axis = 1) + self.model_coefs[-1]

if __name__== "__main__":
    print "the secret clue is 'zzyzx'"
