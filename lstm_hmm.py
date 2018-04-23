#!/usr/bin/env python3

""" 

Modified HMM that is inspired by the psychology behind LSTM-RNNs
    
Designed to eliminate the exponential decay from which HMMs suffer with data sampled at very 
high frequencies and hidden states that change much less frequently compared to the sample rate.     

"""


from hmm import Hmm
import numpy as np
import csv
import time
import glob

import sys        # for sys.argv
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


class LstmHmm (Hmm):
    
    def __init__(s):
        logging.debug('Initializing an LSTM-HMM')
        Hmm.__init__(s) # super.init()
    
    def log_normalize(s, M=None):
        
        for i in range(s.k): # Set self-loops of Transition matrix to 100% (no decay)
            s.T_kk[i,i] = -np.inf         
        
        Hmm.log_normalize(s, M) # super.log_normalize()
        
        for i in range(s.k): # Set self-loops of Transition matrix to 100% (no decay)
            s.T_kk[i,i] = 0 # 0 = log(1)    
    
            
            
class TestLstmHmm (unittest.TestCase):
    
    @classmethod
    def setUpClass(s):
        """ runs once before ALL tests """
        print("\n...........unit testing class LstmHmm..................")
        

    @unittest.skip('')
    def test_lstm(s):
        print('....... testing lstm alone ......')
        
        # Hmm to generate sequences
        hmm_gen = Hmm()
        hmm_gen.initialize_weights(3,4)
        # Pretend World has 3 emotions - [Happy, Sad, Mad]
        # Pretend World has 4 faces - [Neutral, Genuine Smile, Fake Smile, Frown]
        hmm_gen.T_kk = np.log(np.array([1,.2,.2,
                                        .1,1,.5,
                                        .1,.6,1])).reshape([3,3])
        hmm_gen.E_kd = np.log(np.array([.3,.6,.1,.0001,
                                        .2,.0001,.3,.5,
                                        .2,.0001,.1,.7])).reshape([3,4])
        hmm_gen.P_k = np.log(np.array([.6,.3,.1]))
    
        hidden, observed = hmm_gen.generate_sequences(100, 100)
    
        # Display parameters of the generating-Hmm (T_kk diag's are 1.0)
        print('\nGenerated sequences from...')
        hmm_gen.print_percents() 
        print('P = ' + str(hmm_gen.p_X_mat(observed)))
        
        
        lstmHmm = LstmHmm()
        lstmHmm.initialize_weights(3,4)
        #lstmHmm.print_percents()
        
        lstmHmm.X_mat_train = observed
                
        lstmHmm.wrap_em_train_v(n_iter=50, n_init=4)
        
        print('\n\nTrained to...')
        lstmHmm.print_percents()
        

    #@unittest.skip('')
    def test_lstmHmm_vs_hmm(s):
        """
        TODO:
        - Modified always thinks it converges after 1-2 iters probably because the e/m step is 
          not working properly w the modification (I think it actually gets worse instead of better)
        
        """
        print('....... testing lstm Hmm vs regular Hmm ......')
        # Hmm to generate sequences
        hmm_gen = Hmm()
        hmm_gen.initialize_weights(3,4)
        # Pretend World has 3 emotions - [Happy, Sad, Mad]
        # Pretend World has 4 faces - [Neutral, Genuine Smile, Fake Smile, Frown]
        hmm_gen.T_kk = np.log(np.array([1,.2,.2,
                                        .1,1,.5,
                                        .1,.6,1])).reshape([3,3])
        hmm_gen.E_kd = np.log(np.array([.3,.6,.1,.0001,
                                        .2,.0001,.3,.5,
                                        .2,.0001,.1,.7])).reshape([3,4])
        hmm_gen.P_k = np.log(np.array([.6,.3,.1]))
        
        hidden, observed = hmm_gen.generate_sequences(150, 100) # 150 sequences, 100 long each
        
        # Display parameters of the generating-Hmm (T_kk diag's are 1.0)
        print('\nGenerated sequences from...')
        hmm_gen.print_percents()   
        print('P = ' + str(hmm_gen.p_X_mat(observed)))
        
        
        # Check how many sequences contain only 1 hidden state 
        all_one_hidden_state = 0
        for h in hidden:
            if 1 in np.bincount(h)/len(h):
                all_one_hidden_state += 1
        print('{}% of sequences contained only 1 hidden state'.
              format(100 * all_one_hidden_state/float(len(hidden))))

        # Train with LstmHmm (modified m_step)
        print('Training using modified Hmm (LstmHmm)')
        lstm_hmm_train = LstmHmm()
        lstm_hmm_train.initialize_weights(3,4)
        lstm_hmm_train.X_mat_train = observed
        lstm_hmm_train.wrap_em_train_v(n_init=5, n_iter=250)
        
        print('\n\nLstmHmm trained to...')
        lstm_hmm_train.print_percents()
        print('LstmHmm Train Score = {}'.format(lstm_hmm_train.p_X_mat(observed)))
        
        # Train with normal (non-modified) Hmm
        print('Training using normal Hmm')
        hmm_train = Hmm()
        hmm_train.initialize_weights(3,4)
        hmm_train.X_mat_train = observed
        hmm_train.wrap_em_train_v(n_init=5, n_iter=250)
        
        print('\n\nHmm trained to...')
        hmm_train.print_percents()
        print('Hmm Train Score = {}'.format(hmm_train.p_X_mat(observed)))
        
        # Print these two again
        print('\n\nLstmHmm trained to...')
        lstm_hmm_train.print_percents()
        print('LstmHmm Train Score = {}'.format(lstm_hmm_train.p_X_mat(observed)))    
        
        print('\n\nGenerated sequences from...')
        hmm_gen.print_percents()        
        
    
    
#------------------------------------------------------------  
if __name__ == '__main__':    
    unittest.main()