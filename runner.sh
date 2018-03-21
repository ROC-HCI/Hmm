#!/bin/sh
#SBATCH --partition=standard --time=04:00:00
#SBATCH --output=batch_output/output.txt%j
#SBATCH --error=batch_output/error.txt%j
#SBATCH -c 30
#SBATCH --begin now
module load python
module load anaconda
python truth_bluff.py -seed 10 -n 5 -k 4 -i changes_only &
python truth_bluff.py -seed 10 -n 5 -k 5 -i changes_only &
python truth_bluff.py -seed 10 -n 5 -k 6 -i changes_only
