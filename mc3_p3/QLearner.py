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

        # stores real world examples exposed to by the model
        self.real = []
        self.step = 0

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q_tbl[s,])

        if self.verbose: print "s =", s,"a =",action

        self.s = s
        self.a = action

        return action

    def hallucinate(self):

        for i in range(0, self.dyna):

            random = rand.choice(self.real)
            rand_s = random[0]
            rand_a = random[1]

            #dyna_s_prime = self.T_tbl[rand_s, rand_a, :].argmax()
            #print dyna_s_prime
            dyna_s_prime = np.random.multinomial(1, self.T_tbl[rand_s, rand_a, :]).argmax()

            dyna_maxq_prime = np.max(self.Q_tbl[dyna_s_prime,])

            dyna_r = self.R_tbl[rand_s,rand_a]

            #Update Q table with rand_s, rand_a, s_prime, reward
            q_update = ((1-self.alpha) * self.Q_tbl[rand_s, rand_a]) + \
                (self.alpha * (dyna_r + self.gamma * dyna_maxq_prime))

            self.Q_tbl[rand_s, rand_a] = q_update
            #print 'rand_s: ', rand_s, 'rand_a: ', rand_a, 'q_update: ', q_update


    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new state
        @returns: The selected action
        """
        # 1) Updates Tables
        # a) Update Q'[s,a]
        q_state = [self.Q_tbl[s_prime, act] for act in range(0, self.Q_tbl.shape[1])]
        maxq = max(q_state)

        # check if there is more than one action with maxq value, if so, pick one at random
        count = q_state.count(maxq)
        if count > 1:
            rmax = [i for i in range(0,len(q_state)) if q_state[i] == maxq]
            action = rand.choice(rmax)
        else:
            action = np.argmax(q_state)

        q_new = ((1-self.alpha) * self.Q_tbl[self.s, self.a]) + \
                (self.alpha * (r + self.gamma * maxq))
        self.Q_tbl[self.s, self.a] = q_new


        # b) Update T'[s,a,s'] - prob in state s, take action a, will end up in s'
        self.Tcount[self.s, action, s_prime] += 1
        self.T_tbl[self.s, action, :] = self.Tcount[self.s, action, :] / \
                                        self.Tcount[self.s, action, :].sum()

        # c) Update R'[s,a]
        self.R_tbl[self.s, action] = ((1-self.alpha) * self.R_tbl[self.a, action]) + \
                                     (self.alpha * r)

        self.real.append((self.s, action, s_prime, r)) #remember encountered examples to randomize
        self.step +=1

        # d) hallucinate dyna examples, if learner has encountered at least 5 real world examples
        if (self.dyna > 0 and len(self.real) > 5): self.hallucinate()

        # 2) Choose random action with probability self.rar
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions-1)
            if self.verbose: print 'this action is random'


        if self.verbose:
            print "s =", self.s,"a =",action,"r =",r, "s'=", s_prime, "q':", q_new
            print self.Q_tbl

        # 3) update learner values to prime_s and prime_a
        self.s = s_prime
        self.a = action

        # 4) Decay Random Action Rate
        self.rar = self.rar * self.radr # decay rar with radr

        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
