"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.Q_tbl = np.random.uniform(-1.0, 1.0, size=(num_states, num_actions))
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = np.float(alpha)
        self.gamma = np.float(gamma)
        self.rar = np.float(rar)     # probability of selecting random action at each step
        self.radr = np.float(radr)   # random decay rate after each update
        self.dyna = dyna

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        #same logic as query
        self.s = s

        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q_tbl[s,])


        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """
        # 1) Decay Random Action Rate
        self.rar = self.rar * self.radr # decay rar with radr

        # 2) Update Q'[s,a]
        q = [self.Q_tbl[self.s, act] for act in range(0, self.num_actions)]
        maxq = max(q)

        # check if there is more than one action with maxq value, if so, pick one at random
        count = q.count(maxq)
        if count > 1:
            rbest = [i for i in range(0,len(self.num_actions)) if q[i] == maxq]
            action = rand.choice(rbest)
        else:
            action = q.index(maxq)

        value = (1-self.alpha) * r + self.alpha * (r + self.gamma * maxq)
        self.Q_tbl[s_prime, action] = value

        # 3) Choose random action with probability self.rar
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
            print 'this one is random'


        if self.verbose: print "s =", s_prime,"a =",action,"r =",r, "s'=", s_prime, "r':", value

        # 4) update learner values to prime_s and prime_a
        self.s = s_prime
        self.a = action

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
