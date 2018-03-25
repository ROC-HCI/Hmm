# Hidden Markov Models (HMM)

Hidden Markov Model code developed from scratch to be used to identify deceptive behavior from the ROC-HCI deception dataset.

*Authors: Taylan Sen and Matt Levin*


## Main Files
* hmm.py - Main code to implement and test HMM
* lstm_hmm.py - A modified HMM to reduce exponential decay from prolonged hidden states
* truth_bluff.py - Uses KFolds cross-validation on dataset to train two HMMs to determine if test set samples are liars or truth-tellers
* runner.sh - Runs various hyperaparameter permutations to identify best performing models and records results for later analysis
* dispatch.sh - Dispatches permutations of runner.sh to be run on BlueHive
* graphing.py - Used to graph the results from results.csv to find trends