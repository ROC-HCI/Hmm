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

try:
    from scipy.misc  import logsumexp
except ImportError:
    def logsumexp(x,**kwargs):
        return np.log(np.sum(np.exp(x),**kwargs))

import unittest
import time

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

        print("...loading weight file: " + filename)
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
    #---------------------------------------------------------------------------
    


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
        print(hmm)

    def test_parse_weight_file(self):
        print("\n...testing parse_weight_file(...)")
        hmm = Hmm()
        hmm.parse_weight_file('unittest.weights')
        print(hmm)

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
        
