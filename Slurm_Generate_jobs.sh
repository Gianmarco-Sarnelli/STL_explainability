#!/bin/bash
#SBATCH --partition=EPYC                     # Partition name
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
test_names=("simpleconvert_M2M" "simpleconvert_E2M" "simpleconvert_H2M" "simpleconvert_J2M" "simpleconvert_B2M" "simpleconvert_T2M" "simpleconvert_G2M" "simpleconvert_S2M" "simpleconvert_M2E" "simpleconvert_E2E" "simpleconvert_H2E" "simpleconvert_J2E" "simpleconvert_B2E" "simpleconvert_T2E" "simpleconvert_G2E" "simpleconvert_S2E" "simpleconvert_M2H" "simpleconvert_E2H" "simpleconvert_H2H" "simpleconvert_J2H" "simpleconvert_B2H" "simpleconvert_T2H" "simpleconvert_G2H" "simpleconvert_S2H" "simpleconvert_M2J" "simpleconvert_E2J" "simpleconvert_H2J" "simpleconvert_J2J" "simpleconvert_B2J" "simpleconvert_T2J" "simpleconvert_G2J" "simpleconvert_S2J")

for test_name in "${test_names[@]}"; do

    echo "Generating jobs for test: $test_name"

    python3 Generate_jobs.py "$test_name" 5 "yes" "EPYC"

    echo "$test_name jobs are generated"

done

# Clear RUNTHIS.txt first
> RUNTHIS.txt

# Append each generated file to RUNTHIS.txt and then delete it
for test_name in "${test_names[@]}"; do
  cat "generated_${test_name}.txt" >> RUNTHIS.txt
  rm "generated_${test_name}.txt"
done
