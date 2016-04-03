"""
Test a learner.  (c) 2015 Tucker Balch
Modified by Pauline Chow for GaTech ML4T

"""

import numpy as np
import math
import time

import LinRegLearner as lrl
import BagLearner as bl
import KNNLearner as knn
import matplotlib.pyplot as plt

def plot_scatter(plotX, plotY, file_name='test'):
    #create graph in sample comparing predicted and actual
    ax = plt.scatter(plotX, plotY, alpha=0.5, c=['red', 'blue'])
    #plt.show()
    fig = ax.get_figure()
    fig.savefig('Output/%s_knn_comparison.png' % file_name)

if __name__=="__main__":
    inf = open('Data/ripple.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    #print testX.shape
    #print testY.shape

    # create a linear regression learner and train it
    #learner = lrl.LinRegLearner(verbose=True) # create a LinRegLearner
    #learner.addEvidence(trainX, trainY) # train it

    #create a knn learner and train it
    # learner = knn.KNNLearner(k=3, verbose=True) # create a knnLearner
    # learner.addEvidence(trainX, trainY) # train it

    #create bag learner and train it
    # learner = bl.BagLearner(learner=knn.KNNLearner,
    #                         kwargs={"k": 5}, bags=20, boost=False, verbose=False)
    learner = bl.BagLearner(learner=lrl.LinRegLearner, verbose=False)
    learner.addEvidence(trainX, trainY)
    Y = learner.query(testX)

    # evaluate in sample
    predY_train = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY_train) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY_train, y=trainY)
    print "corr: ", c[0,1]
    #create graph in sample comparing predicted and actual
    plot_scatter(predY_train, trainY, 'in_sample')

    # evaluate out of sample
    predY_test = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY_test) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY_test, y=testY)
    print "corr: ", c[0,1]

    #create graph in sample comparing predicted and actual
    plot_scatter(predY_test, testY, 'out_sample')

