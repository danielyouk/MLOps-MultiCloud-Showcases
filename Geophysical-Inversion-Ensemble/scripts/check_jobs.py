# check_jobs.py (Corrected for v2 SDK job listing)

import os
import argparse
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from collections import defaultdict

# Import the model configurations to get the list of all experiments
from train_all_models import MODEL_CONFIGS

def check_all_job_statuses_and_logs():
    """
    Connects to Azure ML using v2 SDK, finds the latest job for each
    experiment, and downloads logs for any failed jobs.
    """
    try:
        # --- 1. Connect to Workspace ---
        print("--> Connecting to Azure ML Workspace...")
        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        print(f"--> Connected to workspace: {ml_client.workspace_name}")

        # --- 2. Get the list of experiment names we care about ---
        target_experiment_names = {f"{key}_azureml" for key in MODEL_CONFIGS.keys()}
        print(f"--> Will look for latest jobs in {len(target_experiment_names)} experiments...")

        # --- 3. List ALL jobs and find the latest one for each of our experiments ---
        print("--> Fetching recent jobs from workspace (this may take a moment)...")
        all_jobs = ml_client.jobs.list()

        latest_jobs = {}
        for job in all_jobs:
            if job.experiment_name in target_experiment_names:
                # If we haven't seen this experiment yet, or if this job is newer, store it.
                if job.experiment_name not in latest_jobs or job.creation_context.created_at > latest_jobs[job.experiment_name].creation_context.created_at:
                    latest_jobs[job.experiment_name] = job
        
        print(f"--> Found {len(latest_jobs)} matching jobs to report on.")

        # --- 4. Loop through the latest jobs we found and check their status ---
        for experiment_name, job in latest_jobs.items():
            print("\n" + "="*80)
            print(f"Experiment: '{experiment_name}'")
            print("="*80)
            
            print(f"  Latest Job Name: {job.display_name}")
            print(f"  Status: {job.status}")
            print(f"  Studio URL: {job.studio_url}")

            if job.status == 'Failed':
                log_dir = f"./outputs/logs/{job.name}"
                print(f"  ⬇️ Status is 'Failed'. Downloading logs to: {log_dir}")
                
                try:
                    # Download all logs for the failed job
                    ml_client.jobs.download(name=job.name, download_path=log_dir)
                    
                    std_log_path = os.path.join(log_dir, "user_logs/std_log.txt")
                    
                    if os.path.exists(std_log_path):
                        print("\n  --- 📄 Contents of std_log.txt ---")
                        with open(std_log_path, 'r') as f:
                            lines = f.readlines()
                            for line in lines[-50:]: # Print last 50 lines
                                print(line.strip())
                        print("  -------------------------------------\n")
                    else:
                        print(f"  Could not find 'std_log.txt' in the downloaded logs at {log_dir}")

                except Exception as e:
                    print(f"  ❌ Could not download logs for job {job.name}. Reason: {e}")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("   Please ensure your 'config.json' is correct and you are logged in via 'az login'.")

if __name__ == "__main__":
    check_all_job_statuses_and_logs()