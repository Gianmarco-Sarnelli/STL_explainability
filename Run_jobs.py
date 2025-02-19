import os
import subprocess
import sys
import argparse
import re
import time

parser = argparse.ArgumentParser(description="Runs the tests for associated with each parameter file in 'job_files/params_..'\n Aguments: --test_name, --tests_num, --SLURM")
parser.add_argument('--test_name', default="default", help="Name of the test")
parser.add_argument('--tests_num', type=int, default=0, help="Number of tests to run. If 0 (default) runs all the available tests")
parser.add_argument('--SLURM', type=bool, default=False, help="Decides if running the SLURM job or running the script directly")

args = parser.parse_args()
test_name = args.test_name
tests_num = args.tests_num
SLURM = args.SLURM
if tests_num == 0:
    stop = False
else:
    stop = True

# Get all files in the job_files directory
files = os.listdir("job_files")

# Starting the execution
print("Starting the execution!")

# Running the scripts/jobs
for file in files:
    if file.startswith(f"params_"):

        #Extract the informations in the file name
        pattern = r"params_(.+?)_(\d+)(?:_done)?\.json"
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
        
        if SLURM: # If this is a SLURM job. Here we want to send at most 4 jobs at the same time
            job_path = os.path.join("job_files", f"slurm_{test_name}_{job_id}.sh")
    
            while True: # Here we continuously try to run a job
                try:# Run squeue command to count running jobs
                    start_safety_time = time.time() # A timer to avoid jobs that take more than 3 hors
                    result = subprocess.run(
                        ['squeue', '-u', '$USER', '-h'],  # -h removes the header
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    running_jobs = len(result.stdout.strip().split('\n'))
                    if result.stdout.strip() == '':  # If no jobs are running
                        running_jobs = 0
                        
                    # If we have less than 4 jobs running, submit new job
                    if running_jobs < 4:
                        try:
                            command = ['sbatch', job_path]
                            subprocess.run(command, check=True)
                            print(f"Submitted job {job_id} while there are {running_jobs} running jobs")
                            tests_num -= 1
                        except subprocess.CalledProcessError as e:
                            print(f"Error submitting job {file}: {e.stderr}")
                        break  # Exit the while loop after successful submission
                    else:
                        time.sleep(30)  # Wait 30 seconds before checking again
                        # Creating an emergency exit:
                        if (time.time()-start_safety_time) > (3 * 60 * 60):
                            print("Three hours have passed, I decide to stop Run_jobs.py")
                            sys.exit(1)                        
                except subprocess.CalledProcessError as e:
                    print(f"Error checking job queue: {e.stderr}")
                    time.sleep(30) # Wait before retrying on the next job
                    break

            if (tests_num == 0) and stop:
                break  # Breaks from the for loop if tests_num becomes 0











            job_path = os.path.join("job_files", f"slurm_{test_name}_{job_id}.sh")
            # Submit the job
            try:
                command = ['sbatch', job_path]
                # Run the command and capture output
                subprocess.run(
                    command,
                    check=True           # Raises CalledProcessError if return code != 0
                )
                print(f"Submitted job {job_id}")

                tests_num -= 1
                if tests_num == 0:
                    break # Breaks from the for loop if tests_num becomes 0
                        # Note that if tests_num started at zero then here it 
                        # would become negative and the loop never breaks
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job {file}: {e.stderr}")

        else: # If not SLURM
            params_file = os.path.join("job_files", f"params_{test_name}_{job_id}.json")
            try:
                save_all = "yes"
                command = ['python3', 'Test_distance_slurm.py', params_file, test_name, save_all]
                # Run the command 
                print(f"Submitted job {job_id} without SLURM")
                subprocess.run(
                    command,
                    check=True           # Raises CalledProcessError if return code != 0
                )
                print(f"Completed job {job_id} without SLURM")

                tests_num -= 1
                if (tests_num == 0) and stop:
                    break  # Breaks from the for loop if tests_num becomes 0
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job {file}: {e.stderr}")        