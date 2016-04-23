This is Python version 2.7
===========================

Author: pchow7
Class: Machine Learning for Trading
Term: Spring 2016
Project: mc3_p2

File for project
------------------------------
1. readme.txt (description of files and execution)
2. util.py (function to download data)
3. preprocess.py (download data, calculate technical features and create visualizations, and output orders for in and out of sample)
4. KNNLearner.py (KNN model learner, output of predicted Y values)
5. marketsim.py (calculates portfolio values with order files and back test graphs)


Order of execution for Project
------------------------------
1. Running "preprocess.py" will complete all necessary calculations and graphs:
    - Change symbol name in either lines 274 or 275 in preprocess.py

2. Not all graphs run at the same time, since I had problems with overlapping lines when multiple graphs were saved at the same time

3. Notes on lines in preprocess to comment in and out to replicate visualizations and outputs. Replace "symbol" with "IBM" or "ML4T-220":

    a) Graphs with normalized price for "symbol", predicted Ys and actual Ys
    - Preprocess.py lines 177-178 (in sample)
    - Preprocess.py lines 179-180 (out sample)

    b) Graph with trading strategy, entry and exits
    - Preprocess.py lines 307 (in sample)
    - Preprocess.py lines 313 (out sample)

    c) Send generated orders to market simulator which will calculate porfolio values and output backtesting graphs
    - Preprocess.py lines 319-320 (in sample)
    - Preprocess.py lines 320-321 (out sample)
