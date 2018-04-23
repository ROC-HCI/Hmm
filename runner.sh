#!/bin/sh
#SBATCH --partition=standard
#SBATCH --time=72:00:00
#SBATCH --output=slurm_output/output.txt%j
#SBATCH --error=slurm_output/error.txt%j
#SBATCH -c 30
#SBATCH --begin now

# Runs a single permutation of parameters (passed via command line) on a compute node.


module load python
module load anaconda


# Parse arguments: k seed model folds infolder d
if [[ $# > 1 ]]; then
    k_value="$1"
    seed="$2"
    model="$3"
    folds="$4"
    infolder="$5"
    d_value="$6"
    if [ "$infolder" == "default" ]; then
	d_value=5
	infolder="/home/mlevin6/Desktop/cluster/cluster_sequences/KM_AU06_r_AU12_r_5/default/every_frame"
    elif [ "$infolder" == "pairwise" ]; then
	d_value=25
	infolder="/home/mlevin6/Desktop/cluster/cluster_sequences/KM_AU06_r_AU12_r_5/pairwise/every_frame"
    fi
    echo "k_value = $k_value"
    echo "seed = $seed"
    echo "model = $model"
    echo "folds = $folds"
    echo "infolder = $infolder"
    echo "d_value = $d_value"
    echo
else
	echo "Please provide proper arguments arguments."
	exit
fi


# Invoke truth_bluff.py with proper arguments
# echo "python truth_bluff.py -k $k_value -m $model -seed $seed -n $folds -i $infolder -d $d_value"
python truth_bluff.py -k "$k_value" -m "$model" -seed "$seed" -n "$folds" -i "$infolder" -d "$d_value"

echo "runner.sh complete."