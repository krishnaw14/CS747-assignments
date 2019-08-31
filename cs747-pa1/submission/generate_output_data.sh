#!/bin/bash
declare -a instance_arr=("../instances/i-1.txt" "../instances/i-2.txt" "../instances/i-3.txt")
declare -a algorithm_arr=("round-robin" "epsilon-greedy" "ucb" "kl-ucb" "thompson-sampling")
declare -a horizon_arr=(50 200 800 3200 12800 51200 204800)
declare -a epsilon_arr=(0.002 0.02 0.2)

for i in "${instance_arr[@]}"
do
	for a in "${algorithm_arr[@]}"
	do
		for h in "${horizon_arr[@]}"
		do 
			seed_counter=0
			while [ $seed_counter -le 49 ]
			do
				
				if [ "$a" = "epsilon-greedy" ]
				then
					for e in "${epsilon-arr[@]}"
					do
						echo "$i" "$a" "$h" "$seed_counter" "$e"
						bash bandit.sh --instance ${i} --algorithm ${a} --randomSeed ${seed_counter} --horizon ${h} --epsilon ${e}  > outputData.txt
					done	
				else
					echo "$i" "$a" "$h" "$seed_counter" "${epsilon_arr[1]}"
					bash bandit.sh --instance ${i} --algorithm ${a} --randomSeed ${seed_counter} --horizon ${h} --epsilon ${epsilon_arr[1]} > outputData.txt
				fi
			((seed_counter++))

			done
		done
	done
done