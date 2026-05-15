import subprocess
import argparse
import sys
import os
def main():
    parser = argparse.ArgumentParser(
        description="Launch N parallel workers for the GNN explanation task."
    )
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        required=True,
        help="The total number of parallel workers to launch."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="A comma-separated list of GPU IDs to use (e.g., '0,1,2,3')."
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=['malicious', 'normal'],
        help="The type of dataset to process ('malicious' or 'normal')."
    )
    args = parser.parse_args()
    num_workers = args.num_workers
    if num_workers <= 0:
        print("Error: Number of workers must be a positive integer.", file=sys.stderr)
        exit(1)
    gpu_ids = [int(gid.strip()) for gid in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Available GPUs for workers: {gpu_ids}")
    worker_script = "auto_explanation.py"
    if not os.path.exists(worker_script):
        print(f"Error: Worker script '{worker_script}' not found.", file=sys.stderr)
        exit(1)
    print(f"--- Starting {num_workers} parallel workers for '{args.dataset_type}' dataset ---")
    processes = []
    for worker_id in range(num_workers):
        assigned_gpu = gpu_ids[worker_id % num_gpus]
        command = [
            sys.executable,
            worker_script,
            "--worker-id", str(worker_id),
            "--total-workers", str(num_workers),
            "--gpu-id", str(assigned_gpu), 
            "--dataset-type", args.dataset_type 
        ]
        print(f"  > Launching Worker {worker_id} on GPU {assigned_gpu} with command: {' '.join(command)}")
        log_file_path = f"worker_{worker_id}_output.log"
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((process, log_file_path))
    print(f"\n--- All {num_workers} workers launched. Waiting for them to complete... ---")
    print("You can monitor the individual worker logs (e.g., worker_0_output.log) for progress.")
    for process, log_file_path in processes:
        process.wait()
        if process.returncode == 0:
            print(f"  > Worker with log '{log_file_path}' finished successfully.")
        else:
            print(f"  > Worker with log '{log_file_path}' finished with an error (return code {process.returncode}). Please check the log for details.")
    print("\n--- All parallel processes have completed. ---")
if __name__ == "__main__":
    main()