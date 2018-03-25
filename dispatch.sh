#!/bin/sh

# Dispatches jobs with different parameters -- runner.sh then calls truth_bluff.py while cycling
# through permutations of the remaining parameters (model, folds, input_folder)

# sbatch runner.sh K-VALUE [SEEDS ... ]

sbatch runner.sh 4 0 101
sbatch runner.sh 5 0 101
sbatch runner.sh 6 0 101

sbatch runner.sh 3 0 101
sbatch runner.sh 7 0 101

sbatch runner.sh 4 2002 30003
sbatch runner.sh 5 2002 30003
sbatch runner.sh 6 2002 30003

sbatch runner.sh 3 2002 30003
sbatch runner.sh 7 2002 30003
