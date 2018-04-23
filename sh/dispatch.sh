#!/bin/sh

# Dispatches SLURM jobs with different parameters -- runner.sh then calls truth_bluff.py

# Parameter permutations to use
seeds=( 0 1111 2222 3333 4444 5555 6666 7777 8888 9999 )
k_values=( 4 5 6 )
input_folders=( default pairwise )
models=( hmm lstmHmm )
folds_values=( 5 )

cd ..

for seed in ${seeds[@]}; do
    for k_value in ${k_values[@]}; do
    	for model in ${models[@]}; do
    		for folds in ${folds_values[@]}; do
    			for infolder in ${input_folders[@]}; do
    				# Set d_value and full infolder name based on infolder shorthand
    				if [ "$infolder" == "default" ]; then
    					d_value=5
    					infolder="/home/mlevin6/Desktop/cluster/cluster_sequences/KM_AU06_r_AU12_r_5/default/every_frame"
    				else
    					d_value=25
    					infolder="/home/mlevin6/Desktop/cluster/cluster_sequences/KM_AU06_r_AU12_r_5/pairwise/every_frame"
    				fi
    				echo "sbatch runner.sh $k_value $seed $model $folds $infolder $d_value"
					sbatch runner.sh "$k_value" "$seed" "$model" "$folds" "$infolder" "$d_value"
    			done
    		done
    	done
    done
done


