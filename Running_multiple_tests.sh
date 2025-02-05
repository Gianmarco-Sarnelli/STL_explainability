#!/bin/bash

# List of test names
test_names=["M2M", "M2E", "M2G", "M2S", "E2M", "E2E", "E2G", "E2S", "B2M", "B2E", "B2B", "B2G", "B2S", "G2M", "G2E", "G2B", "G2G", "G2S", "S2M", "S2E", "S2B", "S2G", "S2S"] 
CHECK_INTERVAL = 600

for test_name in "${test_names[@]}"; do
    while true; do
        num_jobs=$(squeue -u gsarne00 | wc -l)
        num_jobs=$((num_jobs - 1))
        
        if [ $num_jobs -eq 0 ]; then
            echo "Starting job for test: $test_name"

            source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate
            python3 Run_jobs.py --test_name "$test_name" --tests_num 18 --SLURM true
            break
        else
            sleep $CHECK_INTERVAL
        fi
    done
done