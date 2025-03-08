#!/bin/bash

# Generating the local jobs
python3 Generate_jobs.py "local_test_T2M" "1" "yes" "THIN"

# Running the local job
python3 Test_distance.py "job_files/params_local_test_T2M_0.json" "local_test_T2M" "yes"