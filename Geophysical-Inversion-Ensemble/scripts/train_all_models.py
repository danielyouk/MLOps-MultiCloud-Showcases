import os
import sys
import subprocess
import argparse
from datetime import datetime
import yaml

def print_strategy_summary():
    """Print the complete training strategy."""
    print("=" * 80)
    print("🎯 FINAL TRAINING STRATEGY")
    print("=" * 80)
    print(f"📊 Total Models for Experimental Portfolio: {len(MODEL_CONFIGS)}")
    print("\n🔧 Models to be Trained:")
    print("-" * 50)
    for model_key, config in MODEL_CONFIGS.items():
        compute = config.get("compute_target", "default")
        print(f" • {model_key:20s} | Backbone: {config['model_name']:45s} | Compute: {compute}")
    print("\n🎯 Ensemble Strategy:")
    print("-" * 50)
    print("• Train all 16 models in parallel to find the best performers.")
    print("• Select the top 5-10 most diverse models based on validation MAE.")
    print("• Blend the predictions of the selected models for the final submission.")
    print("=" * 80)

def submit_training_job(model_key, config, model_idx, test_mode, auto_confirm, compute_name):
    """
    Submit a single training job to Azure ML using the v2 SDK script.
    
    Args:
        model_key (str): The user-friendly name of the model/experiment.
        config (dict): The configuration dictionary for this model.
        model_idx (int): The unique index (0-15) for this job, used for data splitting.
        test_mode (bool): Flag to run in test mode.
        auto_confirm (bool): Flag to skip confirmation prompts.
        compute_name (str): Name of the Azure ML compute cluster.
    """
    if not auto_confirm:
        print(f"\n🤔 Submit training job for {model_key}?")
        response = input("   Continue? (y/n): ").lower().strip()
        if response != 'y':
            print(f"   ⏭️  Skipping {model_key}")
            return False, ""
    
    print(f"\n🚀 Submitting job for {model_key} (Model Index: {model_idx})...")
    
    python_executable = sys.executable # Assuming venv is activated

    cmd = [
        python_executable, "scripts/run_azureml.py",
        "--model_name", config['model_name'],
        "--learning_rate", str(config['learning_rate']),
        "--batch_size", str(config['batch_size']),
        "--experiment_name", f"{model_key}_azureml",
        "--model_idx", str(model_idx),  # Pass the unique model index
        "--compute_name", compute_name,
        "--num_epochs", str(config['num_epochs']),
        "--early_stopping_patience", str(config['early_stopping_patience']),
        "--early_stopping_min_delta", str(config['early_stopping_min_delta'])
    ]
    if test_mode:
        cmd.append('--test_mode')

    print(f"   Command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ)
        
        if "Job Submission Successful" in process.stdout:
             print(f"   ✅ {model_key} submitted successfully!")
             return True, process.stdout
        else:
            print(f"   ❌ Failed to submit {model_key}: No success message in output.")
            print(f"   Output: {process.stdout}")
            return False, process.stdout

    except subprocess.CalledProcessError as e:
        output = e.stdout + e.stderr
        if "Job Submission Successful" in output:
            print(f"   ✅ {model_key} submitted successfully (ignoring subsequent error).")
            return True, output
        else:
            print(f"   ❌ Failed to submit {model_key}: {e}")
            print(f"   Error output: {e.stderr}")
            return False, e.stderr

def main():
    config_path = 'configs/model_configs.yml'
    print(f"--> Loading model configurations from: {config_path}")
    try:
        with open(config_path, 'r') as file:
            MODEL_CONFIGS = yaml.safe_load(file)
        print(f"--> Found {len(MODEL_CONFIGS)} model configurations.")
    except FileNotFoundError:
        print(f"FATAL ERROR: The config file was not found at '{config_path}'")
        return
    parser = argparse.ArgumentParser(description='Train all models for the ensemble')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with reduced epochs')
    parser.add_argument('--auto_confirm', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_CONFIGS.keys()), 
                        help='Specific models to train (default: all)')
    parser.add_argument('--strategy_only', action='store_true', help='Only show strategy, don\'t train')
    parser.add_argument('--compute_name', type=str, default='my-16-node-cluster', help='Name of the Azure ML compute cluster to use')
    
    args = parser.parse_args()
    
    if args.strategy_only:
        print("\n📋 Strategy summary shown. Use --auto_confirm to start training.")
        return
    
    models_to_train = args.models if args.models else list(MODEL_CONFIGS.keys())
    
    print(f"\n🎯 Training {len(models_to_train)} models on compute cluster '{args.compute_name}'...")
    if args.test_mode:
        print("🧪 TEST MODE: This will pass the --test_mode flag to the Azure ML job.")
    
    # Track results
    successful_jobs = []
    failed_jobs = []
    
    # --- MODIFIED SUBMISSION LOOP ---
    # We use enumerate to get a unique index 'i' (0-15) for each model.
    for i, model_key in enumerate(models_to_train):
        if model_key in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_key]
            
            # Use the specific compute_target from the config if it exists,
            # otherwise use the default one passed as an argument.
            target_cluster = config.get("compute_target", args.compute_name)
            
            print(f"--> Preparing job for '{model_key}' on compute target '{target_cluster}'")
            
            success, output = submit_training_job(
                model_key=model_key, 
                config=config, 
                model_idx=i,
                test_mode=args.test_mode, 
                auto_confirm=args.auto_confirm,
                compute_name=target_cluster 
            )
            if success:
                successful_jobs.append(model_key)
            else:
                failed_jobs.append(model_key)
        else:
            print(f"❌ Unknown model: {model_key}")
            failed_jobs.append(model_key)
    
    print("\n" + "=" * 80)
    print("📊 SUBMISSION SUMMARY")
    print("=" * 80)
    print(f"Total jobs submitted: {len(models_to_train)}")
    print(f"✅ Successful jobs: {len(successful_jobs)}")
    print(f"❌ Failed jobs: {len(failed_jobs)}")
    
    if successful_jobs:
        print(f"\n✅ Successfully submitted:")
        for job in successful_jobs:
            print(f"   • {job}")
    
    if failed_jobs:
        print(f"\n❌ Failed to submit:")
        for job in failed_jobs:
            print(f"   • {job}")
    
    print("\n🎯 Next Steps:")
    print("1. Monitor the jobs in the Azure ML Studio.")
    print("2. Once complete, download the trained models.")
    print("3. Run inference on the hold-out test set to get predictions for each model.")
    print("4. Use your optimization notebook to find the best ensemble weights.")
    print("=" * 80)

if __name__ == '__main__':
    main() 