# Hmm - Hidden Markov Models

Hidden Markov Model code developed from scratch to be used to identify deceptive behavior from the ROC-HCI deception dataset. 

*Authors: Taylan Sen and Matt Levin*


## Files
* hmm.py - Main code to implenet and test HMM
* lstm_hmm.py - A modified HMM to reduce exponential decay from prolonged hidden states
* truth_bluff.py - Uses KFolds cross-validation on dataset to train two HMMs to determine if unlabeled samples are liars or truth-tellers
* runner.sh - Runs various hyperaparameter permutations to identify best performing models and records results for later analysis
* test.sh - Testing script for use during development
