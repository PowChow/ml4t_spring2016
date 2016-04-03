"""
Generates dataset for linear regression
"""

import numpy as np
import csv

def main():
    np.random.seed(125)

    #X_val = np.random.normal(2, 2, 1000)
    X_val = np.linspace(0, 10, 1000)
    X_val_2 = np.random.normal(loc=1, scale=.2, size = 1000)

    beta = 3 + 0.4* np.random.normal(loc=1, scale=0.3)
    delta = np.random.uniform(-.1, .1, size=(1000,))
    Y_val = np.zeros((len(X_val), 1000))
    #print delta[4]

    for i in range(0, len(X_val)-1):
        #Y_val[:,i] = m[i] * X_val[i] + b[i]
        #Y_val[:, i] = (X_val * beta) + delta[i]
        Y_val[:, i] = X_val + delta[i]

    Y_val = np.round(Y_val, decimals =4)


    with open('./Data/best4linreg.csv', 'w') as f:
        w = csv.writer(f, delimiter=',')
        for i in range(0, X_val.shape[0]):
            w.writerow([X_val[i], X_val_2[i], Y_val[i][0]])

if __name__== "__main__":
    main()