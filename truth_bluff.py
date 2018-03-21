#!/usr/bin/env python

import argparse
from hmm import Hmm
from lstm_hmm import LstmHmm
import numpy as np
import random
import csv
import time
import os
from multiprocessing import Pool
from sklearn.model_selection import KFold


"""
TODO:


 - DONE Change 'writer.writerow(' for cross-val to match results.csv
 - Change runner.sbatch to use full infolder as args.i not the shortcut name (use test.sh for dev)
 - DONE Decide on how to average the test set accuracy accross the folds 
 - DONE Make a unique folder name to dump the weights of the 2 HMM's of each fold
 - DONE Make it work with an LSTM-HMM somehow (probably add an argument and some if's)
 - d is hardcoded to 5 --> Parse it from the args.i (infolder)?

"""


# Wrapper for KFold validation (parallelized)
def cross_validate(args):
    # Run cross validation with config from args
    
    if(args.m == 'lstmHmm'): # Use modified HMM
	truthHmm = LstmHmm()
	bluffHmm = LstmHmm()
    else: # Use normal HMM
	truthHmm = Hmm()
	bluffHmm = Hmm()
	
    truthHmm.read_sequences(args.i + '/truthers')
    bluffHmm.read_sequences(args.i + '/bluffers')
    
    if(len(truthHmm.X_mat_train) == 0 or len(bluffHmm.X_mat_train) == 0):
	raise OSError('No data found, make sure {} contains truthers/bluffers folders'
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
        #print('Train = ', trainSet)
        #print('Test = ', testSet)   
        bluffSets.append([trainSet,testSet])
    
    # Folder to put the weight files in for later analysis
    result_folder = str(time.time()).replace('.', '')
    try:
	os.mkdir('results')
    except OSError: # Already exists
	pass
    os.mkdir('results/' + result_folder)
    
    # Set up the arguments
    func_args = []
    for i in range(len(truthSets)):
        func_args.append([args, truthSets[i], bluffSets[i], i + 1, result_folder])
    
    # Run them all in parallel 
    p = Pool(args.n)
    results = p.map(train_test, func_args)
    
    # Write results to a csv for later graphing/analysis
    with open('results.csv', 'a+') as f:
        writer = csv.writer(f)
	
	# Calculate averages across the Folds
	avg_truth_score = 0.0
	avg_bluff_score = 0.0
	avg_accuracy = 0.0
	total_correct = 0
	total_tested = 0
	avg_t_correct = 0.0
	avg_b_correct = 0.0
	
        for result in results:
	    correct, test_size, truth_score, bluff_score, t_correct, b_correct = result
	    avg_truth_score += truth_score
	    avg_bluff_score += bluff_score
	    avg_accuracy += float(correct) / test_size
	    total_correct += correct
	    total_tested += test_size
	    avg_t_correct += t_correct
	    avg_b_correct += b_correct
	
	avg_accuracy /= args.n
	avg_truth_score /= args.n
	avg_bluff_score /= args.n
	avg_t_correct /= args.n
	avg_b_correct /= args.n
        
        # Writes Result to CSV as: 
	# [Time, k, d, n_init, n_iter, seed, n_folds, total_correct, out_of, oercent_correct,
	#  train_score_T, train_score_B, avg_correct_T, avg_correct_B, model, infolder, result_folder]           
        writer.writerow([time.ctime(),args.k,'5',args.n_init,args.n_iter,args.seed,\
                         args.n, total_correct, total_tested, avg_accuracy * 100, \
	                 avg_truth_score, avg_bluff_score, avg_t_correct * 100, avg_b_correct * 100,\
	                 args.m, args.i, result_folder])          
        
    
# A single fold in the KFold validation
def train_test(args):
    #  Parameters  #
    # args[0] is normal args
    # args[1] is [truthTrainSequences, truthTestSequences]
    # args[2] is [bluffTrainSequences, bluffTestSequences]
    # args[3] is the fold number
    # args[4] is the folder name to dump the weights into
    n_init = args[0].n_init  # Random initializations to try
    n_iter = args[0].n_iter  # Iterations in each initialization
    k = args[0].k # Hidden States
    d = 5 # Outputs (number of clusters used) 
    
    if(args[0].m == 'lstmHmm'):
	truthHmm = LstmHmm()
	bluffHmm = LstmHmm()
    else:
	truthHmm = Hmm()
	bluffHmm = Hmm()
    
    truthHmm.X_mat_train = args[1][0]
    truthHmm.X_mat_test = args[1][1]
    bluffHmm.X_mat_train = args[2][0]
    bluffHmm.X_mat_test = args[2][1]
    testSize = len(truthHmm.X_mat_test) + len(bluffHmm.X_mat_test)
   
    print('# Truth Training Sequences: {0}\n# Bluff Training Sequences: {1}'.format(\
        len(truthHmm.X_mat_train), len(bluffHmm.X_mat_train)))
    print('k = {0}, d = {1}, n_init = {2}, n_iter = {3}, testSize = {4}'.format(\
        k,d,n_init,n_iter,testSize))   
    
    print('Beginning training on Truth-Tellers....')
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
        print('Trained truthHmm #',i+1,' Score = ',score)
    # Rebuild the best truthHmm
    truthHmm.P_k, truthHmm.T_kk, truthHmm.E_kd = bestWeights
    
    print('Best Trained Truth-Tellers HMM:')
    truthHmm.print_percents()
    
    print('Beginning training on Bluffers....')        
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
        print('Trained bluffHmm #',i+1,' Score = ',score)
    # Rebuild the best bluffHMM
    bluffHmm.P_k, bluffHmm.T_kk, bluffHmm.E_kd = bestWeights    
    
    print('\nBest Trained Truth-Tellers HMM:')
    truthHmm.print_percents()
    print('\nBest Trained Liars HMM:')
    bluffHmm.print_percents()
    
    # Evaluate on Testing sequences
    correct = 0
    t_correct, b_correct = 0, 0
    for X in truthHmm.X_mat_test:
        if(truthHmm.p_X(X) > bluffHmm.p_X(X)):
            correct += 1
	    t_correct += 1
    for X in bluffHmm.X_mat_test:
        if(bluffHmm.p_X(X) > truthHmm.p_X(X)):
            correct += 1
	    b_correct += 1
    
    print('Out of {0} test cases, {1} were correctly classified.'.format(\
        testSize, correct))
    
    # Train Score
    truthScore = truthHmm.p_X_mat(truthHmm.X_mat_train)
    bluffScore = bluffHmm.p_X_mat(bluffHmm.X_mat_train)
    
    # Write weight files for later usage
    truthHmm.write_weight_file('results/{}/truthers_fold_{}.weights'.format(args[4], args[3]))
    bluffHmm.write_weight_file('results/{}/bluffers_fold_{}.weights'.format(args[4], args[3]))
    # Write results of this fold and human-readable percents
    with open('results/{}/results_fold_{}.txt'.format(args[4], args[3]), 'w+') as f:
	out = 'Out of {0} test cases, {1} were correctly classified'.format(\
	    testSize, correct)
	out += '\nt_correct = {}\nb_correct = {}\ntrain_score_T = {}\ntrain_score_B = {}\n\n'.format(
	     t_correct, b_correct, truthScore, bluffScore)
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
    return correct, testSize, truthScore, bluffScore, t_correct, b_correct

#------------------------------------------------------------------------
def run(args):
    """ 
    [WITHOUT CROSS-VALIDATION]
    Trains an HMM on Truth-Tellers and one on Bluffers then 
        writes the weight files to truthers.weights and bluffers.weights
        Uses test data to test classification (if proper HMM scores higher) """
    print('\n...Testing truthers vs bluffers...')
    
    #  Parameters  #
    n_init = args.n_init  # Random initializations to try
    n_iter = args.n_iter # Iterations in each initialization
    k = args.k # Hidden States
    d = 5 # Outputs (number of clusters used)
    testSize = args.testSize # Number seq's to be used in each X_mat_test and not in training
    # If testSize is 15 for example, 15 truthers and 15 blufers will be withheld from training
    seed = args.seed # Random seed so we can recreate runs (for Test vs Train data)
    
    truthHmm = Hmm()
    bluffHmm = Hmm()

    truthHmm.read_sequences(args.i + '/truthers')
    bluffHmm.read_sequences(args.i + '/bluffers')
    
    if(len(truthHmm.X_mat_train) == 0 or len(bluffHmm.X_mat_train) == 0):
        raise OSError('ERR: No data found, make sure {} contains truthers/bluffers folders'
	              .format(args.i))
    
    # Separate Test and Train data
    random.seed(seed)
    truthHmm.X_mat_test = []
    bluffHmm.X_mat_test = []        
    for i in range(testSize):
        truthHmm.X_mat_test.append(truthHmm.X_mat_train.pop(random.randrange(len(
            truthHmm.X_mat_train))))
        bluffHmm.X_mat_test.append(bluffHmm.X_mat_train.pop(random.randrange(len(
            bluffHmm.X_mat_train))))          
    
    print('# Truth Training Sequences: {0}\n# Bluff Training Sequences: {1}'.format(\
        len(truthHmm.X_mat_train), len(bluffHmm.X_mat_train)))
    print('k = {0}, d = {1}, n_init = {2}, n_iter = {3}, seed = {4}'.format(\
        k,d,n_init,n_iter,seed))
    
    print('Beginning training on Truth-Tellers....')
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
        print('Trained truthHmm #',i+1,' Score = ',score)
    # Rebuild the best truthHmm
    truthHmm.P_k, truthHmm.T_kk, truthHmm.E_kd = bestWeights
    
    print('Best Trained Truth-Tellers HMM:')
    truthHmm.print_percents()
    
    print('Beginning training on Bluffers....')        
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
        print('Trained bluffHmm #',i+1,' Score = ',score)
    # Rebuild the best bluffHMM
    bluffHmm.P_k, bluffHmm.T_kk, bluffHmm.E_kd = bestWeights    
    
    print('\nBest Trained Truth-Tellers HMM:')
    truthHmm.print_percents()
    print('\nBest Trained Liars HMM:')
    bluffHmm.print_percents()
    
    print('k = {0}, d = {1}, n_init = {2}, n_iter = {3}, seed = {4}'.format(\
        k,d,n_init,n_iter,seed)) # Print this again for convenience
    
    # Write weight files for later usage
    truthHmm.write_weight_file('truthers.weights')
    bluffHmm.write_weight_file('bluffers.weights')
    
    # Evaluate on Testing sequences
    # TODO: Would it be helpful to have a separate counter for truth/bluff?
    correct = 0
    for X in truthHmm.X_mat_test:
        if(truthHmm.p_X(X) > bluffHmm.p_X(X)):
            correct += 1
    for X in bluffHmm.X_mat_test:
        if(bluffHmm.p_X(X) > truthHmm.p_X(X)):
            correct += 1
    
    print('Out of {0} test cases, {1} were correctly classified.'.format(\
        testSize + testSize, correct))
    
    truthScore = truthHmm.p_X_mat(truthHmm.X_mat_train)
    bluffScore = bluffHmm.p_X_mat(bluffHmm.X_mat_train)
    
    # Write results to text file for easy reading
    with open('results.txt', 'a+') as f:
        f.write('\n\n-----\n')
        f.write('k = {0}, d = {1}, n_init = {2}, n_iter = {3}, seed = {4}'.format(\
                k,d,n_init,n_iter,seed))        
        f.write('\nOut of {0} test cases, {1} were correctly classified.'.format(\
        testSize + testSize, correct))
        f.write('\ntruthHmm score on training data = {0}'.format(truthScore))
        f.write('\nbluffHmm score on training data = {0}'.format(bluffScore))
        f.write(args.i + ' used for input sequences.')

    # Write results to a csv for later graphing
    with open('results.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow([time.ctime(),k,d,n_init,n_iter,seed,correct,testSize*2,\
                         testSize,truthScore,bluffScore,100*correct/(testSize*2),args.i])        



#------------------------------------------------------------------------
if __name__ == '__main__':

    # Setup commandline parser
    help_intro = 'Program to train two HMMs and classify testing sequences as truthers/bluffers.' 
    parser = argparse.ArgumentParser(description=help_intro)

    parser.add_argument('-k',help='k (number hidden states), ex:3',\
                        type=int, default=4)
    parser.add_argument('-n_init', help='Number of random initializations used to train each HMM', \
                        type=int, default=3)
    parser.add_argument('-n_iter', help='Number of iterations for each initialization', \
                        type=int, default=400)
    parser.add_argument('-seed', help='Random seed used to select Testing sequences', \
                        type=int, default=15)  
    parser.add_argument('-i', help='Input folder (path)', type=str,
                        default='input_sequences/AU06_AU12_KM_5/default/every_frame')
    parser.add_argument('-n', help='Number of folds of k-fold cross-validation', \
                        type=int, default=5)
    parser.add_argument('-m', help='Model - [lstmHmm or hmm], default=hmm', type=str, default='hmm')
    args = parser.parse_args()
    
    print('args: ', args)

    # run(args) # Run and write to results.txt and results.csv (append)
    cross_validate(args)
    
    print('\nPROGRAM COMPLETE')
