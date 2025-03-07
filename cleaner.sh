#!/bin/bash

# Create directories if they don't exist
mkdir -p Outputs
mkdir -p Errors

# Process output log files
for logfile in output_*.log; do
    # Extract test_name from filename - everything between output_ and _NUMBER.log
    if [[ $logfile =~ output_(.*)_[0-9]+\.log ]]; then
        test_name="${BASH_REMATCH[1]}"
        
        # Append contents to the corresponding test output file
        cat "$logfile" >> "Outputs/${test_name}.log"
        rm "$logfile"
    fi
done

# Process error log files
for logfile in error_*.log; do
    # Extract test_name from filename - everything between error_ and _NUMBER.log
    if [[ $logfile =~ error_(.*)_[0-9]+\.log ]]; then
        test_name="${BASH_REMATCH[1]}"
        
        # Append contents to the corresponding test output file
        cat "$logfile" >> "Errors/${test_name}.log"
        rm "$logfile"
    fi
done

rm Generate_jobs_error.log
rm Generate_jobs_output.log
rm RUNTHIS.txt
rm job_runner_10*
