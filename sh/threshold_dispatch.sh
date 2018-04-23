#!/bin/sh
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --output=output.txt%j
#SBATCH --error=error.txt%j
#SBATCH -c 16
#SBATCH --begin now

module load python
module load anaconda

cd ../analysis

python thresholds.py