import unittest
import numpy as np
import BagLearner as bag
import KNNLearner as knn

from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor


class BagTest(unittest.TestCase):
    def setUp(self):
        self.rip = np.genfromtxt("Data/ripple.csv", delimiter=",")
        np.random.shuffle(self.rip)
        self.slice = np.floor(self.rip.shape[0] * .6)
        train,test = self.rip[:self.slice], self.rip[self.slice:]
        self.xtrain = train[:,0:train.shape[1]-1]
        self.ytrain = train[:,train.shape[1]-1]
        self.xtest = test[:, 0:test.shape[1]-1]
        self.ytest = test[:,test.shape[1]-1]

def t_generator(k, nbags):
    def t(self):
        sbag = BaggingRegressor(KNeighborsRegressor(k), nbags, self.xtrain.shape[0])
        learner = bag.BagLearner(knn.KNNLearner, {'k': k}, nbags, boost=False, verbose=False)
        learner.addEvidence(self.xtrain, self.ytrain)

        sy = sbag.fit(self.xtrain, self.ytrain).predict(self.xtest)
        y = learner.query(self.xtest)
        rmse = np.linalg.norm(sy - y) / np.sqrt(len(sy))
        self.assertLess(rmse, .22)
    return t


for k in np.linspace(1,10,10,dtype=int):
    t = t_generator(k, 20)
    name = "test_k{0}".format(k)
    setattr(BagTest, name, t)

if __name__ == "__main__":
    unittest.main()