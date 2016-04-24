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

        self.Q = {} # track states and actions: generate random uniform values -1.0 and 1.0
                    # how many tuples are expected? # states? np.random.uniform(low=-1.0, high=1.0, size=num_states)

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
        action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """

        #get q value based on s_prime and set of actions, get max q
        q = [self.Q(s_prime, act) for act in self.num_actions]
        maxq = max(q)

        # choose random action with probability self.rar
        if random.random() < self.rar:
            #action = rand.randint(0, self.num_actions-1)\
            #use the range of q values
            minq = min(q)
            mag = max(abs(minq), abs(maxq))

            # attribute random values to all possible actions, now recalculate maxq value
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxq = max(q)

        # check if there is more than one action with maxq value, if so, pick one at random
        count = q.count(maxq)
        if count > 1:
            rbest = [i for i in range(len(self.num_actions)) if q[i] == maxq]
            i = random.choice(rbest)
        else:
            i = q.index(maxq)

        #attribute new action with maxq - either randomly assigned or by looking up in Q Table
        action = self.num_actions[i]
        self.rar = self.rar * self.radr # decay rar with radr

        #update Q table values
        value = r + self.gamma * q
        oldq = self.Q.get((self.s, self.a))

        if oldq is None:
            self.Q[(self.s, self.a, s_prime, action)] = r
        else:
            self.Q[(self.s, self.a, s_prime, action)] = oldq + self.alpha * (value - oldq)

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        # reset "previous" state to "new" values
        self.s = s_prime
        self.a = action

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
