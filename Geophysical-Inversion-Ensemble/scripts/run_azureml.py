# scripts/run_azureml.py (Definitive Final Version)

import argparse
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

def main():
    parser = argparse.ArgumentParser(description="Submit an Azure ML Training Job using v2 SDK")
    
    # Arguments passed from the orchestrator script
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--model_idx", type=int, required=True)
    parser.add_argument("--compute_name", type=str, required=True)
    
    # --- ADDED ARGUMENTS TO CONTROL TRAINING ---
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--early_stopping_patience", type=int, required=True)
    parser.add_argument("--early_stopping_min_delta", type=float, required=True)
    parser.add_argument("--test_mode", action="store_true")
    # --------------------------------------------
    
    args = parser.parse_args()

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    job_inputs = {
        "training_data": Input(type="uri_folder", path="azureml:train-samples-asset:1"),
        "split_files": Input(type="uri_folder", path="azureml:file-splits-asset:1"),
    }

    test_mode_str = " --test_mode" if args.test_mode else ""
    
    # --- The command now correctly passes ALL arguments ---
    job_command = (
        "pip install -r requirements.txt && "
        # Use the correct script name: train_multi_gpu.py
        f"python src/train_multi_gpu.py "
        f"--data_dir ${{inputs.training_data}} "
        f"--split_dir ${{inputs.split_files}} "
        f"--model_name '{args.model_name}' "
        f"--experiment_name {args.experiment_name} "
        f"--batch_size {args.batch_size} "
        f"--learning_rate {args.learning_rate} "
        f"--model_idx {args.model_idx} "
        f"--num_epochs {args.num_epochs} "
        f"--early_stopping_patience {args.early_stopping_patience} "
        f"--early_stopping_min_delta {args.early_stopping_min_delta} "
        f"{test_mode_str}"
    )

    job_environment = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"
    )

    job = command(
        code="./",
        command=job_command,
        inputs=job_inputs,
        environment=job_environment,
        compute=args.compute_name,
        distribution={"type": "pytorch", "process_count_per_instance": 1},
        experiment_name=args.experiment_name,
        display_name=f"{args.experiment_name}__{args.model_name.split('.')[0]}"
    )

    print(f"\nSubmitting job for model: {args.model_name}...")
    returned_job = ml_client.jobs.create_or_update(job)
    print("="*60)
    print(f"Job Submission Successful! View in Studio: {returned_job.studio_url}")
    print("="*60)

if __name__ == "__main__":
    main()