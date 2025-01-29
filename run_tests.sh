#!/bin/bash

for i in {0..53}; do
    echo "Executing iteration $i"
    python3 Run_jobs.py --test_name "New_everything" --tests_num 1 --SLURM true
done
