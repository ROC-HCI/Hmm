#!/bin/sh
#SBATCH --partition=standard
#SBATCH --time=16:00:00
#SBATCH --output=slurm_output/output.txt%j
#SBATCH --error=slurm_output/error.txt%j
#SBATCH -c 30
#SBATCH --begin now

module load python
module load anaconda


# Parameters to use
input_folders=( /home/mlevin6/Desktop/cluster/cluster_sequences/KM_AU06_r_AU12_r_5/default/every_frame )
k_values=( 4 5 6 )
seeds=( 0 101 2002 )
folds_values=( 5 )
models=( hmm lstmHmm )

# Allow runner.sh to be called (by dispatcher.sh) to enqueue multiple jobs
# If 1 CL argument given, use it as the k-value
if [[ $# == 1 ]]; then
    k_values=( "$1" )
fi
# If more than one CL arg given, use first as k-value, rest as seeds
if [[ $# > 1 ]]; then
    k_values=( "$1" )
    shift
    seeds=( "$@" )
    echo "k-value = $k_values"
    echo "seeds = [${seeds[@]}]"
fi

echo

for input_folder in ${input_folders[@]}; do
    for k_value in ${k_values[@]}; do
        for seed in  ${seeds[@]}; do
            for folds in ${folds_values[@]}; do
                for model in ${models[@]}; do
                    echo "python truth_bluff.py -k $k_value -m $model -seed $seed -n $folds -i $input_folder"
                    python truth_bluff.py -k "$k_value" -m "$model" -seed "$seed" -n "$folds" -i "$input_folder"
                done
            done
        done
    done
done


echo "Batch runner.sh complete."