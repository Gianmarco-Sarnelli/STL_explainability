#!/bin/bash

# List of test names
test_names=("nountill_M2M") # "nountill_M2E" "nountill_M2B" "nountill_M2G" "nountill_M2S" "nountill_E2M" "nountill_E2E" "nountill_E2B" "nountill_E2G" "nountill_E2S" "nountill_B2M" "nountill_B2E" "nountill_B2B" "nountill_B2G" "nountill_B2S" "nountill_G2M" "nountill_G2E" "nountill_G2B" "nountill_G2G" "nountill_G2S" "nountill_S2M" "nountill_S2E" "nountill_S2B" "nountill_S2G" "nountill_S2S")                      
CHECK_INTERVAL=60

for test_name in "${test_names[@]}"; do
    while true; do
        num_jobs=$(squeue -u gsarne00 | wc -l)
        num_jobs=$((num_jobs - 1))

        if [ $num_jobs -eq 0 ]; then
            echo "Starting job for test: $test_name"

            source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate
            python3 Generate_jobs.py "$test_name" 18
            python3 Run_jobs.py --test_name "$test_name" --tests_num 18 --SLURM true
            sleep 30

            break
        else
            sleep $CHECK_INTERVAL
        fi
    done
done
