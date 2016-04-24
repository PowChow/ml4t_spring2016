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
        #self.Q_tbl = np.zeros(shape=(num_states, num_actions))

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

        #DYNA initializations
        self.T_tbl = np.ones(shape=(num_states, num_actions, num_states))
        self.T_tbl = self.T_tbl/num_states

        self.Tcount = np.full(shape=(num_states, num_actions, num_states),
                             fill_value=.00001, dtype=float)
        self.R_tbl = np.full(shape=(num_states, num_actions),
                             fill_value=.00001, dtype=float)# prob rewards in state a, take action a

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q_tbl[self.s,])

        if self.verbose: print "s =", s,"a =",action

        self.s = s
        self.a = action

        return action

    def hallucinate(self):

        for d in range(0,self.dyna):
            rand_s = rand.sample(xrange(num_states), 1)
            rand_a = rand.sample(xrange(num_actions), 1)

                # s' = infer from T[]
                # r = R[s,a]
                # Update Q table with hallucinated

            #return -- don't think i will return anything


    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """
        # 1) Decay Random Action Rate
        self.rar = self.rar * self.radr # decay rar with radr

        # 2) Updates Tables
        # 2a) Update Q'[s,a]
        q = [self.Q_tbl[self.s, act] for act in range(0, self.Q_tbl.shape[1])]
        maxq = max(q)

        # check if there is more than one action with maxq value, if so, pick one at random
        count = q.count(maxq)
        if count > 1:
            rmax = [i for i in range(0,len(q)) if q[i] == maxq]
            action = rand.choice(rmax)
        else:
            action = np.argmax(q)

        maxq_prime = np.max(self.Q_tbl[s_prime,])
        a_prime = np.argmax(self.Q_tbl[s_prime,])
        q_new = ((1-self.alpha) * self.Q_tbl[self.s, action]) + \
                (self.alpha * (r + self.gamma * maxq_prime))
        self.Q_tbl[self.s, action] = q_new

        #### ADD IN DYNA Q TABLES AND HALLUCINATIONS
        #2b) Update T'[s,a,s'] - prob in state s, take action a, will end up in s'
        self.Tcount[self.s, action, s_prime] += 1
        self.T_tbl[self.s, action, :] = self.Tcount[self.s, action, :] / \
                                        self.Tcount[self.s, action, :].sum()

        #2c) Update R'[s,a]
        self.R_tbl[self.a, action] = ((1-self.alpha) * self.R_tbl[self.a, action]) + \
                                     (self.alpha * r)

        # 3) Choose random action with probability self.rar
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
            #print 'this one is random'

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r, "s'=", s_prime, "q':", q_new

        # 4) update learner values to prime_s and prime_a
        self.s = s_prime
        self.a = action



        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
