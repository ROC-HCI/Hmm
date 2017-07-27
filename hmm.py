#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
Classes to implement a Hidden Markov Model (HMM). 
-------------------------------------------------------------------------------
"""
import numpy as np
import csv
import time

import sys        # for sys.argv
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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
    
    MIN_LOG_P = -200
    
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
        s.E_kd = np.log(np.random.rand(s.k,s.d))

        # Transition log probability T_kk[cur_state,next_state]
        s.T_kk = np.log(np.random.rand(s.k,s.k))
        
        # Prior log probability P_k[state]
        s.P_k  = np.log(np.random.rand(s.k))
        
        s.log_normalize(s.E_kd)
        s.log_normalize(s.T_kk)
        s.log_normalize(s.P_k)
        
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
                
                elif preamble[0] == 'P':
                    state = int(preamble[1])
                    s.P_k[state] = float(sline[1])
                
                else:  
                    assert(True), "ERROR: unknown weight file entry"  
                    
        s.log_normalize(s.T_kk)
        s.log_normalize(s.E_kd)
        s.log_normalize(s.P_k)        
        

    #---------------------------------------------------------------------
    def write_weight_file(s,filename):
        """
        Writes the current weights to the given filename to be 
        read at a later time
        """
        logging.debug('...writing output weight file: ' + filename)

        with open(filename, 'w') as f:
            # k,d (# States, # Outputs)
            f.write(str(s.k) + ',' + str(s.d) + "\n")
            
            # E (Emission Probabilities)
            for i in range(s.k):
                for j in range(s.d):
                    f.write("E_{0}_{1} {2}\n".format(i,j,s.E_kd[i,j]))
            
            # T (Transition Probabilities)
            for i in range(s.k):
                for j in range(s.k):
                    f.write('T_{0}_{1} {2}\n'.format(i,j,s.T_kk[i,j]))
            
            # P (Prior Probabilities)
            for i in range(s.k):
                f.write("P_{0} {1}\n".format(i,s.P_k[i]))
              
    #---------------------------------------------------------------------------
    def parse_data_file(s,filename):
        """ returns np.array of points """
        logging.debug('parse_data_file: ' + filename)
        with open(filename) as f:
            seq = np.array([int(x) for x in f.read().split()])
            f.close()
        return seq
    
        #with open("unittest.seq1") as f:
            #seq = []
            #for line in f:
                ##sline = line.split(', ')
                #sline = re.findall(r'[^,;\s]+', line)
                #assert(len(sline) == 1) # TODO: handle multidim output 
                #seq.append(int(sline[0]))
        #return np.array(seq)    

    #--------------------------------------------------------------
    def log_normalize(s, M=None):
        """ given an array of log probabilities M, biases all elements equally 
            in order to normalize.
            M should be a 1D or 2D numpy array.
        """
        
        if M is None:
            s.log_normalize(s.P_k)
            s.log_normalize(s.T_kk)
            s.log_normalize(s.E_kd)
        else:
            M[np.isnan(M)] = s.MIN_LOG_P
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
    def p_XZ(s, X_n, Z_n):
        """ return log[P(X,Y)], the log prob of observations X with states Z """
        # make sure input is in index format
            
        pX = np.sum(s.E_kd[Z_n,X_n])
        p = s.p_Z(Z_n) + pX
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
        v_nk[0,:] = s.P_k + s.E_kd[:,X[0]]

        # t > 0
        for t in range(1,len(X)):
            v_t = s.T_kk + s.E_kd[:,X[t]] + v_nk[t-1,:,np.newaxis]
            v_nk[t,:], bt_nk[t,:] = np.amax(v_t,axis=0), np.argmax(v_t,axis=0)
 
        # t = last
        t = len(X) - 1
        v_max, v_max_q = np.amax(v_nk[t,:]), np.argmax(v_nk[t,:])
        
        # backtrace
        Z_n[t] = v_max_q
        for t in range(len(X)-2,-1,-1):
            Z_n[t] = bt_nk[t+1,Z_n[t+1]]

        return Z_n, v_max    
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
        """
            Given observation X, and valid E,T, returns the forward probability
            alpha = a, in matrix form.
    
            a_nk[time t,tag j] = logsum P(o1,o2...ot,qt=j|T,E)
        """
        
        # Initialize
        a_nk = np.zeros((len(X),s.k)) 
        a_current_k = np.zeros(s.k) 
                
        # t = 0
        a_nk[0,:] = s.P_k[:] + s.E_kd[:,X[0]]

        # t > 0
        for t in range(1,len(X)):       
            a_current_k = s.T_kk + s.E_kd[:,X[t]] + a_nk[t-1,:,np.newaxis]
            a_nk[t,:] = logsumexp(a_current_k,axis=0)            

        return a_nk
    
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
                for next_state in range(s.k):
                    b_current_k[next_state] = b_nk[t+1,next_state] + \
                                              s.T_kk[cur_state,next_state] + \
                                              s.E_kd[next_state,X[t+1]]
                b_nk[t,cur_state] = logsumexp(b_current_k)   # the max value

        return b_nk
    
    #---------------------------------------------------------------------------
    def backward_v(s, X):
        """  
            Given observation X, and valid E,T, returns the backward probability
            beta = b.
            b[t,j] = logsum P(ot+1, ot+2... oT |T,E, tth hidden state = j)
            b[t,j] = logsumexp(b[t+1,:] + T[j,:] + E[:,X[t+1]])
        """       
        
        # Initialize
        b_nk = np.zeros((len(X),s.k)) 
        b_current_k = np.zeros(s.k)

        # Starting at T-1 and going backwards...
        for t in range(len(X)-2,-1,-1):
            b_current_k = b_nk[np.newaxis,t+1] + s.T_kk + s.E_kd[np.newaxis,:,X[t+1]]
            b_nk[t] = logsumexp(b_current_k, axis=1)   # the max value

        return b_nk
    
    
    #---------------------------------------------------------------------------
    def e_step(s):
        """ calculate state occupancy count [s.gamma_k] and
                      state transition count [s.eta_kk]
            based on 
            parameters [s.E_kd, s.T_kk, s.P_k] 
            
            sum_counts_Y_k = gamma_k[i] is the expected count of (Y = i)
            sum_counts_T_kk = eta_kk[i,j] is the expected count of i to j trans
        """
        sum_counts_P_k  = np.zeros_like(s.P_k, dtype=float)
        sum_counts_E_kd = np.zeros_like(s.E_kd, dtype=float)
        sum_counts_T_kk = np.zeros_like(s.T_kk, dtype=float)

        # for each sequence in training set
        for X in s.X_mat_train: 
            a_nk = s.forward_v(X)
            #b_nk = s.backward_v(X)            
            b_nk = s.backward_for(X)            
            counts_Y_nk = np.exp(a_nk + b_nk) # convert out of logspace
            sum_counts_P_k += counts_Y_nk[0]
            for t in range(len(X)-1):
                for cur in range(s.k):
                    for to in range(s.k):
                        sum_counts_T_kk[cur,to] += np.exp( \
                                                    a_nk[t][cur] + \
                                                    b_nk[t+1][to] + \
                                                    s.E_kd[to,X[t+1]] + \
                                                    s.T_kk[cur,to])
                for state in range(s.k):
                    sum_counts_E_kd[state,X[t]] += counts_Y_nk[t,state]
            for state in range(s.k):
                sum_counts_E_kd[state,X[-1]] += counts_Y_nk[-1,state]

        return sum_counts_P_k, sum_counts_E_kd, sum_counts_T_kk

    #---------------------------------------------------------------------------
    def e_step_v(s):
        """ vectorized version """
        
        # TODO
        pass
    
    #---------------------------------------------------------------------------
    def m_step(s, counts):
        """ calculate parameters [s.E_kd, s.T_kk, s.P_k] 
            based on 
            responsibilities [s.Resp_nk]
        """
        smoothing_count = np.e**s.MIN_LOG_P
        
        sum_counts_P_k, sum_counts_E_kd, sum_counts_T_kk = counts
        s.P_k = np.log(sum_counts_P_k + smoothing_count)
        s.T_kk = np.log(sum_counts_T_kk + smoothing_count)
        s.E_kd = np.log(sum_counts_E_kd + smoothing_count)
    
        s.log_normalize()
        
    #---------------------------------------------------------------------------
    def em_train(s, n_iter):
        """ train parameters using em algorithm """
        for _ in range(n_iter):
            counts = s.e_step()
            s.m_step(counts)
    
    #---------------------------------------------------------------------------
    def mle_train(s, smoothing_count=None):
        """ Calculates parameters: [s.E_kd, s.T_kk, s.P_k] 
            given:                 [s.X_mat_train, s.Y_mat_train]
            
            Each of the probabilities is determined solely by counts. After 
            counting, probabilities are normalized and converted to log.
        """
        if smoothing_count==None:
            smoothing_count = np.e**s.MIN_LOG_P
            
        assert(len(s.X_mat_train) == len(s.Y_mat_train), \
              "ERROR: bad len(s.Y_mat_train)")
        
        counts_P_k  = np.zeros_like(s.P_k, dtype=float)
        counts_T_kk = np.zeros_like(s.T_kk, dtype=float)
        counts_E_kd = np.zeros_like(s.E_kd, dtype=float)

        counts_P_k  += smoothing_count
        counts_T_kk += smoothing_count
        counts_E_kd += smoothing_count
        
        # Main loop for counting
        for X,Y in zip(s.X_mat_train, s.Y_mat_train):    
            counts_P_k[Y[0]] += 1
            for t in range(len(X)-1):
                counts_T_kk[Y[t],Y[t+1]] += 1
                counts_E_kd[Y[t],X[t]] += 1
            counts_E_kd[Y[-1],X[-1]] += 1           
    
        s.P_k = np.log(counts_P_k)
        s.T_kk = np.log(counts_T_kk)
        s.E_kd = np.log(counts_E_kd)
    
        s.log_normalize(s.P_k) 
        s.log_normalize(s.T_kk)
        s.log_normalize(s.E_kd)              
        
            
#============================================================================
class TestHmm(unittest.TestCase):
    """ Self testing of each method """
    
    @classmethod
    def setUpClass(s):
        """ runs once before ALL tests """
        print("\n...........unit testing class Hmm..................")

    def setUp(s):
        """ runs once before EACH test """
        pass

    @unittest.skip
    def test_init(s):
        print("\n...testing init(...)")
        hmm = Hmm()
        hmm.initialize_weights(2,3)
        hmm.log_normalize(hmm.T_kk)
        hmm.log_normalize(hmm.E_kd)
        hmm.log_normalize(hmm.P_k)
        
        print(hmm)

    @unittest.skip
    def test_parse_weight_file(s):
        print("\n...testing parse_weight_file(...)")
        hmm = Hmm()
        hmm.parse_weight_file('unittest.weights')
        print(hmm)
        
    @unittest.skip
    def test_parse_data_file(s):
        print("\n...testing parse_data_file(...)")
        hmm = Hmm()
        seq = hmm.parse_data_file('unittest.seq1')
        print(seq)
    
    @unittest.skip    
    def test_forward_for(s):
        print("\n...testing forward_for(...)")

        hmm = Hmm()
        hmm.initialize_weights(2,3)
        hmm.P_k = np.log(np.array([1,0]))
        hmm.T_kk = np.log(np.array([[0, 1],[1./3, 2./3]]))
        hmm.E_kd = np.log(np.array([[1./3, 1./3, 1./3], \
                                    [0.50, 0.50, 0.00]]))
        hmm.log_normalize()
    
        X1 = np.array([0])
    
        a_nk1 = hmm.forward_for(X1)
        s.assertAlmostEqual(np.log(1./3),a_nk1[0,0])
        s.assertAlmostEqual(-np.inf,a_nk1[0,1])
        
        X2 = np.array([0,1])
        a_nk2 = hmm.forward_for(X2)
        
        # use brute force to check all permutations for a better score
        Zpermutations = list(itertools.product(range(hmm.k), repeat=len(X2)))
        p_a = []
        for Z in Zpermutations:
            p_a.append(hmm.p_XZ(X2,Z))
        p_tot_brute_force = logsumexp(p_a)
        p_tot_alpha = logsumexp(a_nk2[-1])
        s.assertEqual(p_tot_brute_force, p_tot_alpha)
        
        s.assertAlmostEqual(np.log(1./3), a_nk2[0,0])
        s.assertAlmostEqual(-np.inf,a_nk2[0,1])
        s.assertAlmostEqual(-np.inf, a_nk2[1,0])
        s.assertAlmostEqual(np.log(1./3) + np.log(1./2), a_nk2[1,1])
        
        X3 = np.array([0,1,1])
        a_nk3 = hmm.forward_for(X3)
        s.assertAlmostEqual(np.log(1./3), a_nk3[0,0])
        s.assertAlmostEqual(-np.inf,a_nk3[0,1])
        s.assertAlmostEqual(-np.inf, a_nk3[1,0])
        s.assertAlmostEqual(np.log(1./3) + np.log(1./2), a_nk3[1,1])

        P_010 = hmm.p_XZ(X3,[0,1,0])
        P_011 = hmm.p_XZ(X3,[0,0,0])
        s.assertAlmostEqual(logsumexp([P_010,P_011]), a_nk3[2,0])
        
        hmm.parse_weight_file('unittest.weights')
        hmm.log_normalize()
        
        X_n = hmm.parse_data_file('unittest.seq1')        
        a_nk = hmm.forward_for(X_n)
        print(a_nk)
        # use brute force to check all permutations for a better score
        Zpermutations = list(itertools.product(range(hmm.k), repeat=len(X_n)))
        p_a = []
        for Z in Zpermutations:
            p_a.append(hmm.p_XZ(X_n,Z))
        p_tot = logsumexp(p_a)
         
    @unittest.skip    
    def test_backward_for(s):
        print("\n...testing backward_for(...)")

        hmm = Hmm()
        hmm.initialize_weights(2,3)
        hmm.P_k = np.log(np.array([1,0]))
        hmm.T_kk = np.log(np.array([[0, 1],[1./3, 2./3]]))
        hmm.E_kd = np.log(np.array([[1./3, 1./3, 1./3], \
                                        [0.50, 0.50, 0.00]]))
        hmm.log_normalize()
    
        X1 = np.array([0])
    
        b_nk1 = hmm.backward_for(X1)
        s.assertAlmostEqual(0, b_nk1[0,0])
        s.assertAlmostEqual(0, b_nk1[0,1])
    
        X2 = np.array([0,1])
        b_nk2 = hmm.backward_for(X2)        
        P_01 = np.log(1./2)
        P_00 = -np.inf        
        s.assertAlmostEqual(logsumexp([P_00,P_01]), b_nk2[0,0])
        P_10 = np.log(1./3) + np.log(1./3)
        P_11 = np.log(2./3) + np.log(1./2)        
        s.assertAlmostEqual(logsumexp([P_10,P_11]), b_nk2[0,1])
        s.assertAlmostEqual(0, b_nk2[1,0])
        s.assertAlmostEqual(0, b_nk2[1,1])        
        
        a_nk2 = hmm.forward_for(X2)
        y_nk2 = a_nk2 + b_nk2
        Zpermutations = list(itertools.product(range(hmm.k), repeat=len(X2)))
        p_a = []
        for Z in Zpermutations:
            p_a.append(hmm.p_XZ(X2,Z))
        p_tot_brute_force = logsumexp(p_a)
        
        p_tot_alpha_beta = logsumexp(y_nk2[0])
        s.assertEqual(p_tot_brute_force, p_tot_alpha_beta)        
        
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')        
        print(hmm.backward_for(X_n))
        
    @unittest.skip    
    def test_forward_v(s):
        print("\n...testing forward_v(...)")
        hmm = Hmm() # Set up
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')
        
        # Not vectorized
        t = time.time()
        withFor = hmm.forward_for(X_n)
        unvectorziedTime = time.time() - t
        
        # Vectorized
        t = time.time()
        withV =  hmm.forward_v(X_n)
        vectorziedTime = time.time() - t
        
        print("UnVectorized Time: ", unvectorziedTime)           
        print("Vectorized Time: ", vectorziedTime) 
        print("Speedup = ", unvectorziedTime / vectorziedTime, "x")
        
        if(np.allclose(withV, withFor)): # Check if they're ~equal
            print('Equal! Vectorized and UnVectorized are equal!')
        else:
            print('Not Equal! Vectorized and UnVectorized are not equal!')
            print("V = ", withV)
            print("For = ", withFor)
            s.fail(msg="Failure! test_forward test failed")
            
    # TODO: fix this after the bug fix in backward_for
    def test_backward_v(s):
        print("\n...testing backward_v(...)")
        hmm = Hmm() # Set up
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')
        
        # Not vectorized
        t = time.time()
        withFor = hmm.backward_for(X_n)
        unvectorziedTime = time.time() - t
        
        # Vectorized
        t = time.time()
        withV =  hmm.backward_v(X_n) 
        vectorziedTime = time.time() - t
        
        print("UnVectorized Time: ", unvectorziedTime)           
        print("Vectorized Time: ", vectorziedTime) 
        print("Speedup = ", unvectorziedTime / vectorziedTime, "x")
        
        if(np.allclose(withV, withFor)): # Check if they're ~equal
            print('Equal! Vectorized and UnVectorized are equal!')
        else:
            print('Not Equal! Vectorized and UnVectorized are not equal!')
            print("V = ", withV)
            print("For = ", withFor) 
            s.fail(msg="Failure! test_backward_v failed")
            
    @unittest.skip
    def test_mle_train(s):
        print("\n...testing mle_train(...)")
        hmm = Hmm()
        hmm.initialize_weights(2,3)
        X1 = np.array([0,0,1,1,2])
        Y1 = np.array([0,1,0,1,0])
        hmm.X_mat_train = [X1]
        hmm.Y_mat_train = [Y1]
        hmm.mle_train()
        s.assertTrue(np.allclose(np.e**hmm.P_k, np.array([1,0])))
        s.assertTrue(np.allclose(np.e**hmm.T_kk, np.array([[0, 1],
                                                           [1, 0]])))
        s.assertTrue(np.allclose(np.e**hmm.E_kd, 
                                 np.array([[  1./3, 1./3, 1./3],
                                           [  0.50, 0.50, 0.00]])))
    @unittest.skip
    def test_em_train(s):
        print("\n...testing em_train(...)")
        hmm = Hmm()
        hmm.initialize_weights(3,3)
        #hmm.E_kd = np.array([[0,-200, -200],[-200,0,-200],[-200,-200,0]])
        #hmm.T_kk = np.array([[-200,0,-200],[-200,-200,0],[0,-200, -200]])
        #hmm.P_k = np.array([0,-200,-200])
        X1 = np.array([0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2])
        hmm.X_mat_train = [X1]
        hmm.em_train(50)
        print(hmm)
        
        #TODO: add assert to check that weights ended up in a good place

        
    @unittest.skip
    def test_viterbi_for(s):
        print("\n...testing viterbi_for(...)")
        hmm = Hmm()
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')
        Z_n, v_max = hmm.viterbi_for(X_n)
        p = hmm.p_XZ(X_n,Z_n)
        print("Z:",Z_n, "v_max:",v_max,"p:",p)
        s.assertAlmostEqual(p, v_max, 3)
        
        # use brute force to check all permutations for a better score
        Zpermutations = list(itertools.product(range(hmm.k), repeat=len(X_n)))
        p_max = -np.inf
        for Z in Zpermutations:
            p = hmm.p_XZ(X_n,Z)
            p_max = max(p,p_max)
            print(Z, "\tp:", p)

        s.assertAlmostEqual(p_max, v_max, 3)        

    @unittest.skip
    def test_viterbi_v(s):
        print("\n...testing viterbi_v(...)")
        
        hmm = Hmm() # Set up
        hmm.parse_weight_file('unittest.weights')
        X_n = hmm.parse_data_file('unittest.seq1')
       
        # With For loops
        t = time.time()
        Z_n_for, v_max_for = hmm.viterbi_for(X_n)
        p_for = hmm.p_XZ(X_n,Z_n_for)
        forTime = time.time() - t
       
        # Vectorized version
        t = time.time()
        Z_n, v_max = hmm.viterbi_v(X_n)
        p = hmm.p_XZ(X_n,Z_n)     
        vTime = time.time() - t   

        print("Z_for:",Z_n_for, "v_max_for:",v_max_for,"p_for:",p_for)
        print("Z_vec:",Z_n, "v_max_vec:",v_max,"p_vec:",p)             
        
        print("UnVectorized Time: ", forTime) 
        print("Vectorized Time: ", vTime) 
        print("Speedup = ", forTime / vTime, "x")
        
        if(not all(Z_n == Z_n_for)):
            s.fail(msg="Failure! test_viterbi_v failed")
        s.assertAlmostEqual(p, p_for, places=3,)
        s.assertAlmostEqual(v_max, v_max_for, places=3)
    
    @unittest.skip                 
    def test_write_weights(s):
        print("\n...testing write_weight_file(...)")
        hmm = Hmm() # Set up
        hmm.parse_weight_file('unittest.weights')
        #X_n = hmm.parse_data_file('unittest.seq1')
        for i in range(hmm.k):
            print("P_{0} = {1}".format(i,hmm.P_k[i]))
        
        hmm.write_weight_file('unittest.weights.output')
    
    def tearDown(s):
        """ runs after each test """
        pass
    
    @classmethod
    def tearDownClass(s):
        print("\n...........unit testing of class Hmm complete..............\n")
        


#===============================================================================
if __name__ == '__main__':
    if (len(sys.argv) > 1):        
        unittest.main()
    else:
        unittest.main()
        
