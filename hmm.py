#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
Classes to implement a Hidden Markov Model (HMM). 
-------------------------------------------------------------------------------
"""
import numpy as np
import csv

import sys        # for sys.argv
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random
import re
import itertools # for product

try:
    from scipy.misc  import logsumexp
except ImportError:
    def logsumexp(x,**kwargs):
        return np.log(np.sum(np.exp(x),**kwargs))

import unittest
import logging
logging.basicConfig(level=logging.DEBUG)
'''
you can set the logging level from a command-line option such as:
  --log=INFO
'''

#============================================================================
class Hmm():
    """ Hidden Markov Model
        Transition T and emission E probabilities (weights) are in log format,
        and are fully populated np arrays (not sparse).
        
    """
    
    #---------------------------------------------------------------------------
    def __init__(s): 
        """       
        """
        s.k = 0   # number of states
        s.d = 0   # number of outputs
        
        # Emission log probability E_kd[cur_state,output]
        s.E_kd = np.ones((s.k,s.d), dtype=float) 

        # Transition log probability T_kk[cur_state,next_state]
        s.T_kk = np.ones((s.k,s.k), dtype=float)
        
        # Prior log probability P_k[state]
        s.P_k  = np.ones((s.k), dtype=float)    
        
        s.X_mat_test  = [np.zeros(0)] # list of np.array of output sequences
        s.X_mat_train = [np.zeros(0)] # list of np.array of output sequences

        s.Y_mat_test  = [np.zeros(0)] # list of np.array of state sequences
        s.Y_mat_train = [np.zeros(0)] # list of np.array of state sequences
    
    #---------------------------------------------------------------------------
    def __str__(s):
        out = '------------------------------------\n'
        out += 'hmm data: (#states = ' + str(s.k)
        out += ', #outputs = ' + str(s.d) +')'
        out += '\n------------\nT_kk = \n' + str(s.T_kk)
        out += '\n------------\nE_kd = \n' + str(s.E_kd)
        out += '\n------------\nP_k = \n' + str(s.P_k)
        return out

    #---------------------------------------------------------------------------
    def initialize_weights(s,k_,d_):
        s.k = k_   # number of states
        s.d = d_   # number of outputs
        
        # Emission log probability E_kd[cur_state,output]
        s.E_kd = np.ones((s.k,s.d), dtype=float) * np.log(1./s.d)

        # Transition log probability T_kk[cur_state,next_state]
        s.T_kk = np.ones((s.k,s.k), dtype=float) * np.log(1./s.k)
        
        # Prior log probability P_k[state]
        s.P_k  = np.ones((s.k), dtype=float) * np.log(1./s.k)
        
    #--------------------------------------------------------------
    def parse_weight_file(s,filename):
        """
        T and E parameters loaded from filename. Default value entered
        for weight matrix elements not specified.
        """
        logging.debug('parse_weight_file: ' + filename)

        with open(filename) as f:
            first_line = f.readline()
            first_tokens = first_line.split(',')
            k = int(first_tokens[0])
            d = int(first_tokens[1])
            s.initialize_weights(k,d)
            for line in f:
                sline = line.split(" ")
                preamble = sline[0].split("_")
        
                if preamble[0] == 'T':
                    cur_state  = int(preamble[1])
                    next_state = int(preamble[2])
                    s.T_kk[cur_state,next_state] = float(sline[1]) 
        
                elif preamble[0] == 'E':
                    state  = int(preamble[1])
                    output = int(preamble[2])
                    s.E_kd[state,output] = float(sline[1]) 
                
                else:  
                    assert(True), "ERROR: unknown weight file entry"  
                    
        s.log_normalize(s.T_kk)
        s.log_normalize(s.E_kd)
        s.log_normalize(s.P_k)        
        
    #---------------------------------------------------------------------------
    def parse_data_file(s,filename):
        """ returns np.array of points """
        logging.debug('parse_data_file: ' + filename)
        with open("unittest.seq1") as f:
            seq = []
            for line in f:
                #sline = line.split(', ')
                sline = re.findall(r'[^,;\s]+', line)
                assert(len(sline) == 1) # TODO: handle multidim output 
                seq.append(int(sline[0]))
        return np.array(seq)    

    #--------------------------------------------------------------
    def log_normalize(s, M):
        """ given an array of log probabilities M, biases all elements equally 
            in order to normalize.
            M should be a 1D or 2D numpy array.
        """

        M[np.isnan(M)] = -200
        if(len(M.shape) == 1):
            Z = logsumexp(M)
            M -= Z            
        else:
            # row-wise probabilities must sum to 1 
            # for rowi in range(M.shape[0]):
            #     z= logsumexp(M[rowi,:])
            #     M[rowi,:] -= z            
            Z = logsumexp(M,axis=1) # sum over row, col is wildcard
            M -= Z[:,np.newaxis]    # broadcast row-wise
            
    #------------------------------------------------------------------
    def p_Z(s, Z_n):        
        """ return log[P(Z)], the log probability of the sequence of 
            states Zn.
        """
        from_indices = list(Z_n[:-1]) # concat
        to_indices = Z_n[1:]        
        p = np.sum(s.T_kk[from_indices,to_indices]) + s.P_k[Z_n[0]]
        return p   
    
    #------------------------------------------------------------------
    def p_XZ(self, X_n, Z_n):
        """ return log[P(X,Y)], the log prob of observations X with states Z """
        # make sure input is in index format
            
        pX = np.sum(self.E_kd[Z_n,X_n])
        p = self.p_Z(Z_n) + pX
        return p                                
            
    #---------------------------------------------------------------------------
    def viterbi_for(s, X):
        """
        Given observation X, and valid E,T, computes most likely Z.
        This is the easy to understand version with for loops. It is slower
        than the numpy vectorized version.
        
        We basically scan over time, for each state, calculating the probability
        of coming from each of the possible previous states. (ie. a trellis).
        
        v_nk[t,q] = log P(o1,o2...ot,q1,q2..qt=q|T,E)
   
        selecting the most probable q's as we progress over t
        """
        
        # Initialize
        v_nk = np.zeros((len(X),s.k)) 
        v_current_k = np.zeros(s.k) 
        bt_nk = np.zeros((len(X),s.k),dtype=int) # backtrace table
        Z_n = np.empty(len(X),dtype=int)
        
        # t = 0
        for state in range(s.k):
            v_nk[0,state] = s.P_k[state] + s.E_kd[state,X[0]]

        # t > 0
        for t in range(1,len(X)):
            for cur_state in range(s.k):
                for last_state in range(s.k):
                    v_current_k[last_state] = v_nk[t-1,last_state] + \
                                              s.T_kk[last_state,cur_state] + \
                                              s.E_kd[cur_state,X[t]]
                v_nk[t,cur_state] = np.amax(v_current_k)   # the max value
                bt_nk[t,cur_state] = np.argmax(v_current_k) # index of max value
 
        # t = last
        t = len(X) - 1
        v_max, v_max_q = np.amax(v_nk[t,:]), np.argmax(v_nk[t,:])
        
        # backtrace
        Z_n[t] = v_max_q
        for t in range(len(X)-2,-1,-1):
            Z_n[t] = bt_nk[t+1,Z_n[t+1]]

        return Z_n, v_max
 
    #---------------------------------------------------------------------------
    def viterbi_v(s, X):
        pass
    
    #---------------------------------------------------------------------------
    def forward_for(s, X):
        """
            Given observation X, and valid E,T, returns the forward probability
            alpha = a, in matrix form.
    
            a_nk[time t,tag j] = logsum P(o1,o2...ot,qt=j|T,E)
        """
        
        # Initialize
        a_nk = np.zeros((len(X),s.k)) 
        a_current_k = np.zeros(s.k) 
                
        # t = 0
        for state in range(s.k):
            a_nk[0,state] = s.P_k[state] + s.E_kd[state,X[0]]

        # t > 0
        for t in range(1,len(X)):
            for cur_state in range(s.k):
                for last_state in range(s.k):
                    a_current_k[last_state] = a_nk[t-1,last_state] + \
                                              s.T_kk[last_state,cur_state] + \
                                              s.E_kd[cur_state,X[t]]
                a_nk[t,cur_state] = logsumexp(a_current_k)   # the max value

        return a_nk
    
    #---------------------------------------------------------------------------
    def forward_v(s, X):
        pass
    
    #---------------------------------------------------------------------------
    def backward_for(s, X):
        """  
            Given observation X, and valid E,T, returns the backward probability
            beta = b.
            b[t,j] = logsum P(ot+1, ot+2... oT |T,E, tth hidden state = j)
            b[t,j] = logsumexp(b[t+1,:] + T[j,:] + E[X[t+1],:])
        """       
        
        # Initialize
        b_nk = np.zeros((len(X),s.k)) 
        b_current_k = np.zeros(s.k) 
                
        for t in range(len(X)-2,-1,-1):
            for cur_state in range(s.k):
                for last_state in range(s.k):
                    b_current_k[last_state] = b_nk[t+1,last_state] + \
                                              s.T_kk[last_state,cur_state] + \
                                              s.E_kd[cur_state,X[t+1]]
                b_nk[t,cur_state] = logsumexp(b_current_k)   # the max value

        return b_nk
    
    #---------------------------------------------------------------------------
    def backward_v(s, X):
        pass
    
    
    #---------------------------------------------------------------------------
    def e_step(s):
        """ calculate responsibilities [s.Resp_nk] 
            based on 
            parameters [s.E_kd, s.T_kk, s.P_k] 
        """
        pass
            
        
    
    #---------------------------------------------------------------------------
    def m_step(s):
        """ calculate parameters [s.E_kd, s.T_kk, s.P_k] 
            based on 
            responsibilities [s.Resp_nk]
        """
        pass

    #---------------------------------------------------------------------------
    def em_train(s, n_iter):
        """ train parameters using em algorithm """
        for i in n_iter:
            s.e_step()
            s.m_step()
    
    #---------------------------------------------------------------------------
    def mle_train(s, smoothing_count=0):
        """ Calculates parameters: [s.E_kd, s.T_kk, s.P_k] 
            given:                 [s.X_mat_train, s.Y_mat_train]
            
            Each of the probabilities is determined solely by counts. After 
            counting, probabilities are normalized and converted to log.
        """
    
        assert(s.X_mat_train.shape[0] == s.Y_mat_train.shape[0]), \
              "ERROR: bad s.Y_mat_train.shape[0]" 
        
        self.P_k *= 0
        self.T_kk *= 0
        self.E_kd *= 0

        self.T_kk += smoothing_cnt 
        self.E_kd += smoothing_cnt
        
        # Main loop for counting
        for X,Y in zip(s.train_X_mat, self.train_Y_mat):    
            self.P_k[Y[0]] += 1
            for t in range(len(X)-1):
                self.T_kk[Y[t],Y[t+1]] += 1
                self.E_kd[Y[t],X[t]] += 1
            self.E_kd[Y[-1],X[-1]] += 1           
    
        self.P_k = np.log(self.P_k)
        self.T_kk = np.log(self.T_kk)
        self.E_kd = np.log(self.E_kd)
    
        self.log_normalize(self.P_k) 
        self.log_normalize(self.T_kk)
        self.log_normalize(self.E_kd)              
        
            
#============================================================================
class TestHmm(unittest.TestCase):
    """ Self testing of each method """
    
    @classmethod
    def setUpClass(self):
        """ runs once before ALL tests """
        print("\n...........unit testing class Hmm..................")

    def setUp(self):
        """ runs once before EACH test """
        pass

    @unittest.skip
    def test_init(self):
        print("\n...testing init(...)")
        hmm = Hmm()
        hmm.initialize_weights(2,3)
        hmm.log_normalize(hmm.T_kk)
        hmm.log_normalize(hmm.E_kd)
        hmm.log_normalize(hmm.P_k)
        
        print(hmm)

    @unittest.skip
    def test_parse_weight_file(self):
        print("\n...testing parse_weight_file(...)")
        hmm = Hmm()
        hmm.parse_weight_file('unittest.weights')
        print(hmm)
        
    @unittest.skip
    def test_parse_data_file(self):
        print("\n...testing parse_data_file(...)")
        hmm = Hmm()
        seq = hmm.parse_data_file('unittest.seq1')
        print(seq)

    def test_forward_for(self):
        print("\n...testing forward_for(...)")
        pass

    def test_mle_train(self):
        print("\n...testing m(...)")
        pass


    def test_viterbi_for(self):
        print("\n...testing viterbi_for(...)")
        hmm = Hmm()
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')
        Z_n, v_max = hmm.viterbi_for(X_n)
        p = hmm.p_XZ(X_n,Z_n)
        print("Z:",Z_n, "v_max:",v_max,"p:",p)
        self.assertAlmostEqual(p, v_max, 3)
        #self.assertAlmostEqual(-3.984, v_max, 3)
        
        # use brute force to check all permutations for a better score
        Zpermutations = list(itertools.product(range(hmm.k), repeat=len(X_n)))
        p_max = -np.inf
        for Z in Zpermutations:
            p = hmm.p_XZ(X_n,Z)
            p_max = max(p,p_max)
            print(Z, "\tp:", p)

        self.assertAlmostEqual(p_max, v_max, 3)        

    def tearDown(self):
        """ runs after each test """
        pass
    
    @classmethod
    def tearDownClass(self):
        print("\n...........unit testing of class Hmm complete..............\n")
        


#===============================================================================
if __name__ == '__main__':
    if (len(sys.argv) > 1):        
        unittest.main()
    else:
        unittest.main()
        
