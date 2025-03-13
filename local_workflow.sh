#!/bin/bash

# Generating the local jobs
python3 Generate_jobs.py "local_test_B2M" "1" "yes" "THIN" "Test_model.py"

# Running the local job
python3 Test_model.py "job_files/params_local_test_B2M_0.json" "local_test_B2M" "yes"