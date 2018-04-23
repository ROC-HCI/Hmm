#!/usr/bin/env python

"""
Dual-HMM Classification code using K-Fold cross-validation. Trains one HMM on truth-tellers and
one on bluffers and checks if test sequences fit their respective HMM better than the alternative.

Records results to results.csv

"""

import argparse

from hmm import Hmm
from lstm_hmm import LstmHmm

from csv import writer as csv_writer
from time import time, ctime
from os import mkdir
from sys import exit

import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score as f1

import logging
logging.basicConfig(level=logging.INFO)
# Note: On BlueHive compute nodes the logging goes to stderr not stdout


#------------------------------------------------------------------------
# Wrapper for KFold validation (parallelized)
def cross_validate(args):
    # Run cross validation with config from args
    
    if args.m.lower() in ['lstmhmm', 'lstm']: # Use modified HMM
	truthHmm = LstmHmm()
	bluffHmm = LstmHmm()
    else: # Use normal HMM
	truthHmm = Hmm()
	bluffHmm = Hmm()
	
    truthHmm.read_sequences(args.i + '/truthers')
    bluffHmm.read_sequences(args.i + '/bluffers')
    
    if(len(truthHmm.X_mat_train) == 0 or len(bluffHmm.X_mat_train) == 0):
	raise IOError('No data found, make sure {} contains truthers/bluffers folders'
	        .format(args.i))
     
    # Split the sequences into args.n folds for truth and then bluff
    np.random.seed = args.seed
    kf = KFold(n_splits=args.n)
    
    X = truthHmm.X_mat_train
    truthSets = []
    for train, test in kf.split(X):
        trainSet = []
        testSet = []
        for i in train:
            trainSet.append(X[i])
        for i in test:
            testSet.append(X[i])
        #print('Train = ', trainSet)
        #print('Test = ', testSet)   
        truthSets.append([trainSet,testSet])
    
    X = bluffHmm.X_mat_train
    bluffSets = []
    for train, test in kf.split(X):
        trainSet = []
        testSet = []
        for i in train:
            trainSet.append(X[i])
        for i in test:
            testSet.append(X[i])
        #logging.debug('Train = ', trainSet)
        #logging.debug('Test = ', testSet)   
        bluffSets.append([trainSet,testSet])
    
    # Folder to put the weight files in for later analysis
    result_folder = str(time()).replace('.', '')
    try:
	mkdir('results')
    except OSError: # Already exists
	pass
    mkdir('results/' + result_folder)
    
    # Set up the arguments
    func_args = []
    for i in range(len(truthSets)):
        func_args.append([args, truthSets[i], bluffSets[i], i + 1, result_folder])
    
    # Run them all in parallel 
    p = Pool(args.n)
    try:
	results = p.map(train_test, func_args) # Run folds in parallel
    except (KeyboardInterrupt, Exception):
	logging.error('An error occurred in Pool.')
	p.terminate()
	p.join()
	p.close()
	exit(0)
    finally:
	p.close()
    
    # Write results to a csv for later graphing/analysis
    with open('results.csv', 'a+') as f:
        writer = csv_writer(f)
	
	# Calculate averages across the Folds
	avg_truth_score = 0.0
	avg_bluff_score = 0.0
	avg_accuracy = 0.0
	total_correct = 0
	total_tested = 0
	avg_t_correct = 0.0
	avg_b_correct = 0.0
	avg_f1_score = 0.0
	
        for result in results:
	    correct, test_size, truth_score, bluff_score, t_correct, b_correct, f1_score = result
	    avg_truth_score += truth_score
	    avg_bluff_score += bluff_score
	    avg_accuracy += float(correct) / test_size
	    total_correct += correct
	    total_tested += test_size
	    avg_t_correct += t_correct
	    avg_b_correct += b_correct
	    avg_f1_score += f1_score
	
	avg_accuracy /= args.n
	avg_truth_score /= args.n
	avg_bluff_score /= args.n
	avg_t_correct /= args.n
	avg_b_correct /= args.n
	avg_f1_score /= args.n
        
        # Writes Result to CSV as: 
	# [Time, k, d, n_init, n_iter, seed, n_folds, total_correct, out_of, percent_correct,
	#  train_score_T, train_score_B, avg_correct_T, avg_correct_B, model, infolder, result_folder, f1_score]           
        writer.writerow([ctime(),args.k,args.d,args.n_init,args.n_iter,args.seed,\
                         args.n, total_correct, total_tested, avg_accuracy * 100, \
	                 avg_truth_score, avg_bluff_score, avg_t_correct * 100, avg_b_correct * 100,\
	                 args.m, args.i, 'results/'+result_folder, avg_f1_score])          
        
        
        
#------------------------------------------------------------------------
# A single fold in the KFold validation
def train_test(args):
    #  Parameters  #
    # args[0] is normal args
    # args[1] is [truthTrainSequences, truthTestSequences]
    # args[2] is [bluffTrainSequences, bluffTestSequences]
    # args[3] is the fold number
    # args[4] is the folder name to dump the weights into
    try:
	n_init = args[0].n_init  # Random initializations to try
	n_iter = args[0].n_iter  # Iterations in each initialization
	k = args[0].k # Hidden States
	d = args[0].d # Outputs (number of clusters used) 
	
	if args[0].m.lower() in ['lstmhmm', 'lstm']:
	    truthHmm = LstmHmm()
	    bluffHmm = LstmHmm()
	else:
	    truthHmm = Hmm()
	    bluffHmm = Hmm()
	
	# Assign the train/test sequences for this fold
	# See hmm.Hmm.load_test_sequences for explination on 'wrap_interviews' param
	truthHmm.load_train_sequences(args[1][0])
	truthHmm.load_test_sequences(args[1][1], wrap_interviews=True)
	bluffHmm.load_train_sequences(args[2][0])
	bluffHmm.load_test_sequences(args[2][1], wrap_interviews=True)
	
	testSize = len(truthHmm.X_mat_test) + len(bluffHmm.X_mat_test)
       
	logging.info('# Truth Training Sequences: {0}\n# Bluff Training Sequences: {1}'.format(\
	    len(truthHmm.X_mat_train), len(bluffHmm.X_mat_train)))
	logging.info('k = {0}, d = {1}, n_init = {2}, n_iter = {3}, testSize = {4}'.format(\
	    k,d,n_init,n_iter,testSize))
	
	logging.info('Beginning training on Truth-Tellers....')
	bestScore = -np.inf
	# Run em_train for Truth-Tellers multiple times, finding the best-scoring one
	for i in range(n_init):
	    truthHmm.initialize_weights(k,d)
	    truthHmm.em_train_v(n_iter)
	    score = truthHmm.p_X_mat(truthHmm.X_mat_train)
	    if(score > bestScore):
		bestScore = score
		bestWeights = truthHmm.P_k, truthHmm.T_kk, truthHmm.E_kd
	    truthHmm.print_percents()
	    logging.info('Trained truthHmm #{} Score = {}'.format(i+1,score))
	# Rebuild the best truthHmm
	truthHmm.P_k, truthHmm.T_kk, truthHmm.E_kd = bestWeights
	
	logging.info('Best Trained Truth-Tellers HMM:\n{}'.format(truthHmm.get_percents()))
	
	logging.info('Beginning training on Bluffers....')        
	bestScore = -np.inf # Reset for bluffers
	# Run em_train for Bluffers multiple times, finding the best-scoring one     
	for i in range(n_init):
	    bluffHmm.initialize_weights(k,d)
	    bluffHmm.em_train_v(n_iter)
	    score = bluffHmm.p_X_mat(bluffHmm.X_mat_train)
	    if(score > bestScore):
		bestScore = score
		bestWeights = bluffHmm.P_k, bluffHmm.T_kk, bluffHmm.E_kd
	    bluffHmm.print_percents()
	    logging.info('Trained truthHmm #{} Score = {}'.format(i+1,score))
	# Rebuild the best bluffHMM
	bluffHmm.P_k, bluffHmm.T_kk, bluffHmm.E_kd = bestWeights    
	
	print('\nBest Trained Truth-Tellers HMM:')
	truthHmm.print_percents()
	print('\nBest Trained Liars HMM:')
	bluffHmm.print_percents()
	
	# Evaluate on Testing sequences
	correct = 0 # total classified correctly
	t_correct, b_correct = 0, 0
	# Expected and actual values for F1 Score
	expected = ([0] * len(truthHmm.X_mat_test)) + ([1] * len(bluffHmm.X_mat_test))
	predicted = []	
	# Each X in hmm.X_mat_test is a list, one sequence for each segment of the interview
	# (due to low confidence periods) so they should be evaluated together so each interview
	# is weighted equally.
	for X_interview in truthHmm.X_mat_test:
	    if truthHmm.p_X_mat(X_interview) > bluffHmm.p_X_mat(X_interview):
		correct += 1
		t_correct += 1
		predicted.append(0)
	    else:
		predicted.append(1)
	for X_interview in bluffHmm.X_mat_test:
	    if bluffHmm.p_X_mat(X_interview) > truthHmm.p_X_mat(X_interview):
		correct += 1
		b_correct += 1
		predicted.append(1)
	    else:
		predicted.append(0)
	
	print('Out of {0} test cases, {1} were correctly classified.'.format(\
	    testSize, correct))
	
	# Train Score
	truthScore = truthHmm.p_X_mat(truthHmm.X_mat_train)
	bluffScore = bluffHmm.p_X_mat(bluffHmm.X_mat_train)
	
	# F1 Score
	f1_score = f1(expected, predicted)
	
	# Write weight files for later usage
	truthHmm.write_weight_file('results/{}/truthers_fold_{}.weights'.format(args[4], args[3]))
	bluffHmm.write_weight_file('results/{}/bluffers_fold_{}.weights'.format(args[4], args[3]))
	# Write results of this fold and human-readable percents
	with open('results/{}/results_fold_{}.txt'.format(args[4], args[3]), 'w+') as f:
	    out = 'Out of {0} test cases, {1} were correctly classified'.format(\
		testSize, correct)
	    out += '\nt_correct = {}\nb_correct = {}\ntrain_score_T = {}\ntrain_score_B = {}\n\n'.format(
		 t_correct, b_correct, truthScore, bluffScore)
	    out += 'f1_score = {}\n\n'.format(f1_score)
	    out += '\n\nTruth HMM:\n'
	    out += truthHmm.get_percents()
	    out += '\n\nBluff HMM:\n'
	    out += bluffHmm.get_percents()
	    f.write(out)
	    f.close()
	    
	# Convert to percents for later averaging
	t_correct /= float(len(truthHmm.X_mat_test))
	b_correct /= float(len(bluffHmm.X_mat_test))  
	
	# Return the number correct, testSize to be averaged and written to CSV
	return correct, testSize, truthScore, bluffScore, t_correct, b_correct, f1_score
    
    except KeyboardInterrupt:
	return 'KeyboardInterrupt'



#------------------------------------------------------------------------
if __name__ == '__main__':

    # Setup commandline parser
    help_intro = 'Program to train two HMMs and classify testing sequences as truthers/bluffers.' 
    parser = argparse.ArgumentParser(description=help_intro)

    parser.add_argument('-k',help='k (number hidden states), default:4',\
                        type=int, default=4)
    parser.add_argument('-d',help='d (number outputs / observations possibilities), default:5',\
                        type=int, default=5)    
    parser.add_argument('-n_init', help='Number of random initializations used to train each HMM', \
                        type=int, default=5)
    parser.add_argument('-n_iter', help='Number of iterations for each initialization', \
                        type=int, default=400)
    parser.add_argument('-seed', help='Random seed used to select Testing sequences', \
                        type=int, default=0)  
    parser.add_argument('-i', help='Input folder (path)', type=str,
                        default='input_sequences/subset_data')
    parser.add_argument('-n', help='Number of folds of k-fold cross-validation', \
                        type=int, default=5)
    parser.add_argument('-m', help='Model - [lstmHmm or hmm], default=hmm', type=str, default='hmm')
    args = parser.parse_args()
    
    print('Args:')
    for arg, val in args.__dict__.items():
	print(' {} = {}'.format(arg, val))

    # Run and append results to results.txt and results.csv 
    cross_validate(args)
    
    print('\nPROGRAM COMPLETE')
