#!/bin/sh

# Dispatches SLURM jobs with different parameters -- runner.sh then calls truth_bluff.py

# Parameter permutations to use
#seeds=( 0 1111 2222 3333 4444 5555 6666 7777 8888 9999 )
seeds=( 0 3333 6666 9999 )
k_values=( 4 5 6 )
input_folders=( default )
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
    					d_value=9
    					infolder="/home/mtran14/Desktop/Hmm/gaze_seq_skipfail/subset_data"
    				else
				# will not be executed code
    					d_value=25
    					infolder="/home/mtran14/Desktop/Hmm/gaze_seq_skipfail/subset_data"
    				fi
    				echo "sbatch runner.sh $k_value $seed $model $folds $infolder $d_value"
					sbatch runner.sh "$k_value" "$seed" "$model" "$folds" "$infolder" "$d_value"
    			done
    		done
    	done
    done
done


