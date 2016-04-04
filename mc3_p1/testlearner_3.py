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

def plot_scatter(plotX, plotY, file_name='test', model='knn'):
    #create graph in sample comparing predicted and actual
    ax = plt.scatter(plotX, plotY, alpha=0.5, c=['red', 'blue'])
    #plt.show()
    fig = ax.get_figure()
    fig.savefig('Output/%s_%s_comparison.png' % (file_name, model) )

def plot_data(plotX, plotY, name='ripple'):
    ax = plt.scatter(plotX, plotY, alpha=0.5, c=['red', 'blue'])
    #plt.show()
    fig = ax.get_figure()
    fig.savefig('Output/%s_data_scatter.png' % name )

def plot_line_graphs(plotX, plotY_in, plotY_out, file_name='test'):
    plt.plot(plotX, plotY_in)
    plt.plot(plotX, plotY_out)
    plt.title('Ripple Dataset: KNN with Bagging Overfitting')
    plt.legend(['in_sample', 'out_sample'])
    plt.xlabel('Bags')
    plt.ylabel('Error')

    plt.show()
    plt.figure()
    plt.savefig('Output/%s_error.png' % (file_name) )
    #fig = ax.get_figure()
    #fig.savefig('Output/%s_%s_error.png' % (file_name) )

if __name__=="__main__":
    #inf = open('Data/ripple.csv')
    #inf = open('Data/best4linreg.csv')
    #inf = open('Data/best4knn.csv')
    #inf = open('Data/simple.csv')
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

    in_rmse_k = []
    in_corr_k = []
    out_rmse_k = []
    out_corr_k = []
    model = 'knn' #bag
    print 'shape of datatset:', data.shape
    for i in range(2, 51):
        #learner = knn.KNNLearner(k=i, verbose=True) # create a knnLearner
        learner = bl.BagLearner(learner=knn.KNNLearner,
                                kwargs={"k": 4}, bags=i, boost=False, verbose=False)
        learner.addEvidence(trainX, trainY) # train it

        predY_train = learner.query(trainX) # get the predictions

        #get in sample stats
        in_rmse_k.append(math.sqrt(((trainY - predY_train) ** 2).sum()/trainY.shape[0]))
        c_in = np.corrcoef(predY_train, y=trainY)
        in_corr_k.append(c_in[0,1])

        # get out of sample stats
        predY_test = learner.query(testX) # get the predictions
        out_rmse_k.append(math.sqrt(((testY - predY_test) ** 2).sum()/testY.shape[0]))
        c_out = np.corrcoef(predY_test, y=testY)
        out_corr_k.append(c_out[0,1])


    plot_line_graphs(np.arange(2,51), in_rmse_k, out_rmse_k, 'knn_ripple_4_bags')
