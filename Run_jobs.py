import os
import subprocess
import sys
import argparse
import re
import time

# NOTE: We don't use this anymore for testing on slurm!! check Simple_Job_Runner.sh

parser = argparse.ArgumentParser(description="Runs the tests for associated with each parameter file in 'job_files/params_..'\n Aguments: --test_name, --tests_num, --SLURM")
parser.add_argument('--test_name', default="default", help="Name of the test")
parser.add_argument('--tests_num', type=int, default=0, help="Number of tests to run. If 0 (default) runs all the available tests")
parser.add_argument('--SLURM', type=bool, default=False, help="Decides if running the SLURM job or running the script directly")
parser.add_argument('--iteration', type=int, default=0, help="Number of iteration of the Slurm_job_runner. It's needed only for the slurm execution")

args = parser.parse_args()
test_name = args.test_name
tests_num = args.tests_num
SLURM = args.SLURM
iteration_number = args.iteration
if tests_num == 0:
    stop = False
else:
    stop = True

# Get all files in the job_files directory
files = os.listdir("job_files")

# Path of the Slurm_Job_runner slurm script:
Slurm_Job_runner_path = "Slurm_Job_runner.sh"

# Maximum time for slurm job:
Time_for_SLURM = 1.5 * 60 * 60  # 1.30h

# Starting the execution
print("Starting the execution!")
start_time = time.time()

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
                        ['squeue', '-u', 'gsarne00', '-h'],  # -h removes the header
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
                            time.sleep(30)
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
                
            total_time = time.time() - start_time
            if (time.time() - start_time) > Time_for_SLURM:
                # If it runs out of time we sbatch again the slurm job and break out of the for loop
                try:
                    command = ['sbatch', Slurm_Job_runner_path, f"{iteration_number}"]
                    subprocess.run(command, check=True)
                    print(f"Sbatching again Slurm_Job_runner, time passed: {total_time}")
                except subprocess.CalledProcessError as e:
                    print(f"Error while rerunnig the Slurm job: {e.stderr}")
                    break # Break out of the for loop





        else: # If not SLURM
            params_file = os.path.join("job_files", f"params_{test_name}_{job_id}.json")
            try:
                save_all = "yes"
                command = ['python3', 'Test_distance.py', params_file, test_name, save_all]
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

# At the end of the for loop this script sbatches again "Slurm_job_runner" if SLURM = True
if SLURM:
    try:
        command = ['sbatch', Slurm_Job_runner_path, f"{iteration_number + 1}"]
        subprocess.run(command, check=True)
        print(f"Sbatching Slurm_Job_runner for the next iteration, time passed: {total_time}")
    except subprocess.CalledProcessError as e:
        print(f"Error while rerunnig the Slurm job: {e.stderr}")
