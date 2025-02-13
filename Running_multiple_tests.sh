#!/bin/bash

# List of test names
test_names=("precomp_M2B" "precomp_E2B")
CHECK_INTERVAL=60

for test_name in "${test_names[@]}"; do
    while true; do
        num_jobs=$(squeue -u gsarne00 | wc -l)
        num_jobs=$((num_jobs - 1))

        if [ $num_jobs -eq 0 ]; then
            echo "Starting job for test: $test_name"

            source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate
            python3 Generate_jobs.py "$test_name" 20 "yes"
            python3 Run_jobs.py --test_name "$test_name" --tests_num 20 --SLURM true
            sleep 30

            break
        else
            sleep $CHECK_INTERVAL
        fi
    done
done
