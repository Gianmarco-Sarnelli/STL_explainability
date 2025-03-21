#!/bin/bash
#SBATCH --partition=THIN                     # Partition name
#SBATCH --account=dssc                       # Account name
#SBATCH --ntasks=1                           # Number of tasks (for mpi, not needed now)
#SBATCH --nodes=1                            # Use exactly one node
#SBATCH --cpus-per-task=1                    # CPUs per task (for multiprocessing)
#SBATCH --mem-per-cpu=2G                     # Memory per CPU
#SBATCH --time=2:00:00                       # Time limit (2 hours)
#SBATCH --output=Generate_jobs_output.log    # Standard output log
#SBATCH --error=Generate_jobs_error.log      # Standard error log
#SBATCH --get-user-env                       # Activating the environment


# Activate the virtual environment
source /u/dssc/gsarne00/Environments/expl_orfeo/bin/activate

# List of test names #M E H J B T G S
test_names=("selfnorm_again_M2M")
for test_name in "${test_names[@]}"; do

    echo "Generating jobs for test: $test_name"

    python3 Generate_jobs.py "$test_name" 5 "yes" "THIN" "Test_model.py"

    echo "$test_name jobs are generated"

done

# Clear RUNTHIS.txt first
> RUNTHIS.txt

# Append each generated file to RUNTHIS.txt and then delete it
for test_name in "${test_names[@]}"; do
  cat "generated_${test_name}.txt" >> RUNTHIS.txt
  rm "generated_${test_name}.txt"
done
