import os
import subprocess
import sys
import argparse
import re

parser = argparse.ArgumentParser(description="Runs the tests for associated with each parameter file in 'job_files/params_..'\n Aguments: --test_name, --tests_num, --SLURM")
parser.add_argument('--test_name', default="default", help="Name of the test")
parser.add_argument('--tests_num', type=int, default=0, help="Number of tests to run. If 0 (default) runs all the available tests")
parser.add_argument('--SLURM', type=bool, default=False, help="Decides if running the SLURM job or running the script directly")

args = parser.parse_args()
test_name = args.test_name
tests_num = args.tests_num
SLURM = args.SLURM

# Get all files in the job_files directory
files = os.listdir("job_files")

# Starting the execution
print("Starting the execution!")

# Running the scripts/jobs
for file in files:
    if file.startswith(f"slurm_{test_name}"):

        #Extract the informations in the file name
        pattern = r"slurm_(.+?)_(\d+)(?:_done)?\.sh"
        match = re.match(pattern, file)
        if match:
            test_name_file, job_id = match.groups()
            done = "_done" in file
        else:
            print(f"Match not valid")
            continue
        
        # Skipping if the job is of a different test
        if test_name != test_name_file:
            continue

        # Skipping the job if it's already done
        if done:
            continue
        
        if SLURM: # If this is a SLURM job
            job_path = os.path.join("job_files", file)
            # Submit the job
            try:
                command = ['sbatch', job_path]
                # Run the command and capture output
                print(f"Submitted job {file}")
                result = subprocess.run(
                    command,
                    check=True           # Raises CalledProcessError if return code != 0
                )
                print(f"Completed job {file}")

                
                # Create the new filename by inserting '_done' before the extension
                base = job_path.rsplit('.', 1)[0]  # Get everything before the .sh
                new_filename = f"{base}_done.sh"
                # Rename the file
                os.rename(job_path, new_filename)

                tests_num -= 1
                if tests_num == 0:
                    break # Breaks from the for loop if tests_num becomes 0
                        # Note that if tests_num started at zero then here it 
                        # would become negative and the loop never breaks
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job {file}: {e.stderr}")
        else:
            params_file = f"job_files/params_{test_name}_{job_id}.json"
            job_path = os.path.join("job_files", file)
            try:
                command = ['python3', 'Test_distance_slurm.py', params_file, test_name]
                # Run the command and capture output
                print(f"Submitted job {file} without SLURM")
                subprocess.run(
                    command,
                    check=True           # Raises CalledProcessError if return code != 0
                )
                print(f"Completed job {file} without SLURM")

                # Create the new filename by inserting '_done' before the extension
                base = job_path.rsplit('.', 1)[0]  # Get everything before the .sh
                new_filename = f"{base}_done.sh"
                # Rename the file

                os.rename(job_path, new_filename)

                tests_num -= 1
                if tests_num == 0:
                    break # Breaks from the for loop if tests_num becomes 0
                        # Note that if tests_num started at zero then here it 
                        # would become negative and the loop never breaks
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job {file}: {e.stderr}")
        