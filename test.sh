#!/bin/sh
#SBATCH --partition=standard --time=01:00:00
#SBATCH --output=slurm_output/output.txt%j
#SBATCH -c 30
#SBATCH --begin now

module load python
module load anaconda

python truth_bluff.py -i input_sequences/subset_data -m lstmHmm -k 4 -n_init 3 -n_iter 40 -n 5 -seed 0
python truth_bluff.py -i input_sequences/subset_data -m hmm -k 4 -n_init 3 -n_iter 40 -n 5 -seed 0

rm *.pyc