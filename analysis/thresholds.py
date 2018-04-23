#!/usr/bin/env python

"""
Threshold analysis code - Parses the Dual-HMM classification results from truth_bluff.py and
applies confidence thresholds to analyze the performance on the most-confident classifications only. 

Varies the threshold from 100% (all sequences included) to top 10% only, decrementing by 10% each step.

Records results to threshold_results.csv for later analysis (threshold_analysis.ipynb)

"""

from sys import path, argv, exit
path.append('..')
from hmm import Hmm
from lstm_hmm import LstmHmm

import numpy as np
import pandas as pd

from csv import writer as csv_writer
from os import makedirs, remove
from glob import glob
from time import ctime
import heapq # Priority heap data structure

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score as f1

import logging
logging.basicConfig(level=logging.INFO)
# Note: On BlueHive compute nodes the logging goes to stderr not stdout

# Thresholds - top 100%, 90%, 80%, etc.
THRESHOLD_VALUES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

#------------------------------------------------------------------------
def analyze_row(row):
    """ Analyzes a single row of results.csv (from truth_bluff.py) to see what percent of 
    the most confident test sequences were classified correctly. A sequence is confident
    if there is a large difference between the probability of it being from the truth HMM 
    and the bluff HMM. """
    
    # ---- Extract variables from row ----
    infolder =  row['infolder']
    seed = row['seed']
    k = row['k']
    n_folds = row['n_folds']
    model = row['model']
    result_folder = row['result_folder']
    percent_correct_total = row['percent_correct']
    train_score_T = row['train_score_T']
    train_score_B = row['train_score_B']
    
    # ---- Initialize HMMs with proper model and read input sequences ----
    if model.lower() in ['lstmhmm', 'lstm']: # Use modified HMM
        truthHmm = LstmHmm()
        bluffHmm = LstmHmm()
    else: # Use normal HMM
        truthHmm = Hmm()
        bluffHmm = Hmm()

    truthHmm.read_sequences(infolder + '/truthers')
    bluffHmm.read_sequences(infolder + '/bluffers')

    if(len(truthHmm.X_mat_train) == 0 or len(bluffHmm.X_mat_train) == 0):
        print(infolder)
        raise IOError('No data found, make sure {} contains truthers/bluffers folders'
                      .format(infolder))


    # ------ Split into folds same way as truth_bluff.py ------
    np.random.seed = seed
    kf = KFold(n_splits=n_folds)

    X = truthHmm.X_mat_train
    truthSets = []
    for train, test in kf.split(X):
        trainSet = []
        testSet = []
        for i in train:
            trainSet.append(X[i])
        for i in test:
            testSet.append(X[i]) 
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
        bluffSets.append([trainSet,testSet])

        
    # --- Find the classification accuracy at each threshold, includes 1.0 as baseline ---
    threshold_accuracies = {} # The classification accuracy at each threshold
    threshold_accuraciesT = {} # Truth-tellers only
    folds_with_zero_totalT = {} # Number of folds at a threshold that only had bluffers (ignore in avg)
    threshold_accuraciesB = {} # Bluffers only
    folds_with_zero_totalB = {} # Number of folds at a threshold that only had truth-tellers
    threshold_actual, threshold_predict = {}, {} # Used for F1 Score
    # Probabilities and log probabilities for T/B for train/test sets
    log_p_train_T = []
    log_p_train_B = []
    log_p_test_T = []
    log_p_test_B = []
    
    for threshold in THRESHOLD_VALUES:
        threshold_accuracies[threshold] = 0.0
        threshold_accuraciesT[threshold] = 0.0
        folds_with_zero_totalT[threshold] = 0
        threshold_accuraciesB[threshold] = 0.0
        folds_with_zero_totalB[threshold] = 0
        threshold_actual[threshold] = []
        threshold_predict[threshold] = []
        # Make directory for the outfolder for the sequences at this threshold
        outfolder = 'threshold_{}/top_{}_percent'.format(result_folder, int(threshold * 100))
        try: 
            makedirs(outfolder)
        except OSError as er: # Folder already exists
            print(er)
            print('Removing all files from ' + outfolder)
            map(lambda f: remove(f), glob(outfolder + '/*'))


    # -- For each fold, find the confidence/correctness of the classification of each test seq --
    for i in range(n_folds):
        print('Processing fold #{}...'.format(i+1))
                
        # Reload the HMM weights for T/B HMMs for this fold
        truthHmm.parse_weight_file('../' + result_folder + '/truthers_fold_{}.weights'.format(i+1))
        bluffHmm.parse_weight_file('../' + result_folder + '/bluffers_fold_{}.weights'.format(i+1))
        
        # Get the T/B test sets from this fold
        trainSetT, trainSetB = truthSets[i][0], bluffSets[i][0]
        testSetT, testSetB = truthSets[i][1], bluffSets[i][1]
        
        # Append to p_X_mat and log_p_X_mat for this fold
        for X in trainSetT:
            log_p = truthHmm.p_X(X)
            log_p_train_T.append(log_p)
        for X in trainSetB:
            log_p = truthHmm.p_X(X)
            log_p_train_B.append(log_p)
        for X in testSetT:
            log_p = truthHmm.p_X(X)
            log_p_test_T.append(log_p)
        for X in testSetB:
            log_p = truthHmm.p_X(X)
            log_p_test_B.append(log_p)
        
        # Heap data structure - Entry: (confidence, sequence, isCorrect, T/B)
        confidenceHeap = []
        
        # Add these sequences to the confidence heap       
        for X in testSetT:
            p_T = truthHmm.p_X(X)
            p_B = bluffHmm.p_X(X)
            confidence = abs(p_T - p_B)
            correct = p_T > p_B
            heapq.heappush(confidenceHeap, (confidence, X, correct, 'T'))
        for X in testSetB:
            p_T = truthHmm.p_X(X)
            p_B = bluffHmm.p_X(X)
            confidence = abs(p_T - p_B)
            correct = p_B > p_T
            heapq.heappush(confidenceHeap, (confidence, X, correct, 'B'))
            
        # Calculate accuracy at each confidence threshold for this fold's test set
        for threshold in THRESHOLD_VALUES:
            roundedCount = int(len(confidenceHeap) * threshold)
            correct = 0
            actual, predict = [], []
            correctT, correctB, totalT, totalB = 0,0,0,0
            outfolder = 'threshold_{}/top_{}_percent/'.format(result_folder, int(threshold * 100))
            #print(outfolder)
            for j, X in enumerate(heapq.nlargest(roundedCount, confidenceHeap)):
                """
                For each test sequence that meets the current confidence threshold, see if it is marked
                correct and update tallys (overall and corresponding T/B) and write sequence as a file 
                to outfolder with analysis information (fold number, sequence number, T/B, correct/wrong). 
                """
                if X[2]: # Correct classification
                    correct += 1 # Total correct at this threshold (T + B)
                    
                    if X[3] == 'T':
                        correctT += 1
                        totalT += 1
                        actual.append(0)
                        predict.append(0)
                        open(outfolder + 'fold{}-sequence{}-T-correct.seq'.format(i+1,j), 'w+').write(X[1])
                    else:
                        correctB += 1
                        totalB += 1
                        actual.append(1)
                        predict.append(1)                        
                        open(outfolder + 'fold{}-sequence{}-B-correct.seq'.format(i+1,j), 'w+').write(X[1])
                                            
                else: # Incorrect classification
                    if X[3] == 'T':
                        totalT += 1
                        actual.append(0)
                        predict.append(1)                        
                        open(outfolder + 'fold{}-sequence{}-T-wrong.seq'.format(i+1,j), 'w+').write(X[1])
                    else:
                        totalB += 1
                        actual.append(1)
                        predict.append(0)                        
                        open(outfolder + 'fold{}-sequence{}-B-wrong.seq'.format(i+1,j), 'w+').write(X[1])
                        
                    
            # Accuracy for this fold
            percent_correct = 100.0 * float(correct) / float(roundedCount)
            threshold_accuracies[threshold] += percent_correct
            
            # Accuracy for T/B isolated - If no test sequences in set, ignore this fold from average
            if totalT == 0:
                folds_with_zero_totalT[threshold] += 1
            else:
                percent_correctT = 100.0 * float(correctT) / float(totalT)
                threshold_accuraciesT[threshold] += percent_correctT
            if totalB == 0:
                folds_with_zero_totalB[threshold] += 1
            else:
                percent_correctB = 100.0 * float(correctB) / float(totalB)
                threshold_accuraciesB[threshold] += percent_correctB
                
            threshold_actual[threshold] += actual
            threshold_predict[threshold] += predict

            #.logging.debug('threshold_accuracies[{}] += {}\nNow = {} at fold {}'.
                  #format(threshold, percent_correct, threshold_accuracies[threshold], i+1))
            
            # -- Done processing this fold --
        
        
    f1_scores = {}
    # --- Done with all folds, print results to stdout ---
    for threshold in THRESHOLD_VALUES:
        # Divide by n_folds first since threshold_accuracies is the average accross all folds        
        threshold_accuracies[threshold] /= float(n_folds)
        print('Threshold = {} --> Accuracy = {}%'.format(threshold, threshold_accuracies[threshold]))
        f1_scores[threshold] = f1(threshold_actual[threshold], threshold_predict[threshold])
        print('                   F1 Score = {}%'.format(f1_scores[threshold]))
    for threshold in THRESHOLD_VALUES:
        # Divide by (n_folds) - (# folds with zero totalT at this threshold)
        # So that folds that had no data to classify are ignored in the averaging 
        if folds_with_zero_totalT[threshold] == n_folds:
            threshold_accuraciesT[threshold] = None # This should rarely/never happen, but is possible
        else:
            threshold_accuraciesT[threshold] /= float(n_folds - folds_with_zero_totalT[threshold])
        print('Threshold = {} --> AccuracyT = {}%'.format(threshold, threshold_accuraciesT[threshold]))
    for threshold in THRESHOLD_VALUES:
        # Divide by (n_folds) - (# folds with zero totalT at this threshold)
        # So that folds that had no data to classify are ignored in the averaging         
        if folds_with_zero_totalB[threshold] == n_folds:
            threshold_accuraciesB[threshold] = None # This should rarely/never happen, but is possible
        else:
            threshold_accuraciesB[threshold] /= float(n_folds - folds_with_zero_totalB[threshold])        
        print('Threshold = {} --> AccuracyB = {}%'.format(threshold, threshold_accuraciesB[threshold]))
        
        
    # Check the 1.0 threshold (all sequences included) matches original percent_correct from results.csv
    assert np.isclose(threshold_accuracies[1.0],percent_correct_total),\
           "Threshold of 100% doesn't match original"
    
    # Convert logged probabilities to np.array
    log_p_train_T = np.array(log_p_train_T)
    log_p_train_B = np.array(log_p_train_B)
    log_p_test_T = np.array(log_p_test_T)
    log_p_test_B = np.array(log_p_test_B)
    
    # Calculate non-logged probabilities
    p_train_T = np.exp(log_p_train_T)
    p_train_B = np.exp(log_p_train_B)
    p_test_T = np.exp(log_p_test_T)
    p_test_B = np.exp(log_p_test_B)    
    
    # Check logged train scores matches original from results.csv (The csv value is average of 5 folds)
    assert np.isclose(np.sum(log_p_train_T), train_score_T * 5),\
     "Log Train Score T doesn't match! Got {}, Expected {}".format(np.sum(log_p_train_T), train_score_T * 5)
    # Note: Assert only works for T, not B, because of the way the averaging is calculated since some folds of
    #       of bluffers have a different number of sequences, but all are the same for truth-tellers.
    
    
    # --- Write results to threshold_results.csv for later analysis ---
    f = open('threshold_results.csv', 'a+')
    writer = csv_writer(f)
    row = [ctime(), k, model, infolder, seed, result_folder, n_folds]
    for threshold in THRESHOLD_VALUES:
        row.append(threshold_accuracies[threshold])
    row.append(threshold_accuracies[0.1] > threshold_accuracies[1.0]) # 10% better than 100%?
    for threshold in THRESHOLD_VALUES:
        row.append(threshold_accuraciesT[threshold])
    row.append(threshold_accuraciesT[0.1] > threshold_accuraciesT[1.0]) # 10% better than 100% T?   
    for threshold in THRESHOLD_VALUES:
        row.append(threshold_accuraciesB[threshold])
    row.append(threshold_accuraciesB[0.1] > threshold_accuraciesB[1.0]) # 10% better than 100% B?
    for threshold in THRESHOLD_VALUES:
        row.append(f1_scores[threshold])
    # Write columns for average p_X_mat and log_p_X_mat for test/train and T/B
    row += list(map(lambda ar: np.mean(ar), [log_p_train_T, log_p_train_B, log_p_test_T, log_p_test_B]))
    row += list(map(lambda ar: np.mean(ar), [p_train_T, p_train_B, p_test_T, p_test_B]))
    # And their standard deviations
    row += list(map(lambda ar: np.std(ar), [log_p_train_T, log_p_train_B, log_p_test_T, log_p_test_B]))
    row += list(map(lambda ar: np.std(ar), [p_train_T, p_train_B, p_test_T, p_test_B]))
    
    """
    Entry in format:  [ time,k,model,infolder,seed,result_folder,n_folds,
                        top_100_percent,top_90_percent,top_80_percent,......,10_better_100,
                        top_100_percentT,top_90_percentT,top_80_percentT,...,10_better_100T,
                        top_100_percentB,top_90_percentB,top_80_percentB,...,10_better_100B,
                        top_100_f1_score,top_90_f1_score,top_80_f1_score,...,top_10_f1_score,
                        log_p_train_T, log_p_train_B, log_p_test_T, log_p_test_B,
                        p_train_T, p_train_B, p_test_T, p_test_B,
                        std_log_p_train_T, std_log_p_train_B, std_log_p_test_T, std_log_p_test_B,
                        std_p_train_T, std_p_train_B, std_p_test_T, std_p_test_B ]
    """
    writer.writerow(row)
    f.close()
    print('Wrote entry to threshold_results.csv')



#------------------------------------------------------------------------
if __name__ == '__main__':
    """ Reads best performing rows from truth_bluff_results.csv and calls analyze_row """
        
    np.set_printoptions(threshold=np.inf)  # Don't insert '...' when writing sequences to files

    print('Reading truth_bluff_results.csv...')
    df = pd.read_csv('truth_bluff_results.csv', 
                     usecols=['infolder', 'seed', 'k', 'n_folds', 'model', 'percent_correct',
                              'result_folder', 'train_score_T', 'train_score_B'])
    
    
    if len(argv) > 1 and argv[1] == '-db':
        # Debug mode - just run on first 3 rows
        df = df.iloc[0:3,:]
        print('\nDebug mode - running on first 3 rows only...')
              
    for i, row in df.iterrows():
        
        print('\n--- Analyzing row #{}: ---\n{}\n'.format(i,row))
        
        try:
            analyze_row(row)
        except AssertionError as er:
            # These shouldn't happen
            logging.error('\n{}: {}\nRow = {}\n\n'.format(type(er).__name__, er.message, row)) # Log
            print('{}: {}'.format(type(er).__name__, er.message)) # Print for debugging
        except IOError as er:
            # Just a 'file already exists' error so print for if debugging but don't log for deploying
            print('{}: {}'.format(type(er).__name__, er))            
        
        
    print('\n\nthresholds.py complete\n')
    