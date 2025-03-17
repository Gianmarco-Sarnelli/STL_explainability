#!/bin/bash
#SBATCH --partition=lovelace  # Updated partition name
#SBATCH --account=ai-lab      # Updated account name
#SBATCH --ntasks=1            # Number of tasks
#SBATCH --nodes=1             # Use exactly one node
#SBATCH --cpus-per-task=16    # Increased CPUs for faster job generation
#SBATCH --mem-per-cpu=32G     # Increased memory
#SBATCH --output=Generate_jobs_output.log # Standard output with job ID
#SBATCH --error=Generate_jobs_error.log   # Standard error with job ID
#SBATCH --get-user-env        # Activating the environment


# List of test names #M E H J B T G S
test_names=("selfnorm_pure_M2M" "selfnorm_pure_B2M" "selfnorm_pure_G2M" "selfnorm_pure_H2M" "selfnorm_pure_T2M" )

for test_name in "${test_names[@]}"; do

    echo "Generating jobs for test: $test_name"

    python3 Generate_jobs.py "$test_name" 5 "yes" "lovelace" "Test_model.py"

    echo "$test_name jobs are generated"

done

# Clear RUNTHIS.txt first
> RUNTHIS.txt

# Append each generated file to RUNTHIS.txt and then delete it
for test_name in "${test_names[@]}"; do
  cat "generated_${test_name}.txt" >> RUNTHIS.txt
  rm "generated_${test_name}.txt"
done