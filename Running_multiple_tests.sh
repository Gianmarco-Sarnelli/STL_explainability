#!/bin/bash

# List of test names
test_names=("focusMU0_M2B" "focusMU0_E2B")
CHECK_INTERVAL=60

for test_name in "${test_names[@]}"; do
    while true; do
        num_jobs=$(squeue -u gsarne00 | wc -l)
        num_jobs=$((num_jobs - 1))

        if [ $num_jobs -eq 0 ]; then
            echo "Generating jobs for test: $test_name"

            source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate
            #python3 Generate_jobs.py "$test_name" 20 "yes"
            sbatch Slurm_Generate_jobs.sh "$test_name" 60 "yes"            
            # Additional wait to ensure file system sync
            sleep 30
            break
        else
            sleep $CHECK_INTERVAL
         fi
    done
done

for test_name in "${test_names[@]}"; do
    while true; do
        num_jobs=$(squeue -u gsarne00 | wc -l)
        num_jobs=$((num_jobs - 1))

        if [ $num_jobs -eq 0 ]; then
            echo "Starting job for test: $test_name"
        
            source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate
            
            python3 Run_jobs.py --test_name "$test_name" --tests_num 60 --SLURM true
            sleep 30
                        
            break
        else
            sleep $CHECK_INTERVAL
        fi
    done
done
