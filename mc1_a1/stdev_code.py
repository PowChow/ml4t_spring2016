#!/usr/bin/python
# coding:utf-8

import math as m

"""Spring 2016, OMSCS GaTech - Assignment 1: Standard Deviation"""

___author___  = "Pauline Chow"
___email___ = "pchow@gatech.edu"


def stdev_p(data):
    """Calculates and returns the population standard dev"""
    result = []
    mean_p = sum(data) / len(data)
    [result.append(m.pow(d-mean_p,2)) for d in data]

    return m.sqrt(sum(result) / len(result))


def stdev_s(data):
    """ Calculate sample standard deviation, uses "Bessel Correction"""
    result = []
    mean_p = sum(data) / len(data)
    [result.append(m.pow(d-mean_p,2)) for d in data]

    return m.sqrt(sum(result) / (len(result)-1))

if __name__ == "__main__":
    test = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    print "the population stdev is", stdev_p(test)
    print "the sample stdev is", stdev_s(test)