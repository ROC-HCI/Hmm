# Hidden Markov Models (HMM)

Hidden Markov Model code developed from scratch to be used to identify deceptive behavior from the ROC-HCI deception dataset.

*Authors: Taylan Sen and Matt Levin*


### Main Files/Folders
* hmm.py - Implements and test a hidden Markov model
* truth_bluff.py - Dual-HMM classification on a dataset using cross-validation
* lstm_hmm.py - Modified HMM to avoid exponential decay of prolonged hidden states
* runner.sh - Used to deploy code on a BlueHive computing cluster (SLURM system)
* analysis - Jupyter notebooks to analyze results and code to apply confidence-based thresholds to results
* sh - Shell scripts for dispatching jobs to obtain the dataset and other utilities