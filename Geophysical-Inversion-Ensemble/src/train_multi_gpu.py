# src/train_multi_gpu.py (Definitive Final Version)

import os
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import mlflow
import shutil

# This assumes your other project files are correctly structured
from timm_unet import TimmUnet
from data_utils import load_specific_split
from dataset import SeismicDatasetFileLevel
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


# --- Early Stopping Class ---
class EarlyStopping:
    """
    Early stopping to stop training when validation loss does not improve
    by a minimum ratio for a given number of patience epochs.
    """
    def __init__(self, patience=10, min_improvement_ratio=0.01):
        self.patience = patience
        self.min_improvement_ratio = min_improvement_ratio
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        improvement = (self.best_loss - val_loss) / self.best_loss if self.best_loss != float('inf') else float('inf')

        if improvement > self.min_improvement_ratio:
            self.best_loss = val_loss
            self.counter = 0
            return True # Indicates a new best score was found
        else:
            self.counter += 1
            print(f"--> No significant improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("🛑 Early stopping triggered!")
                self.early_stop = True
            return False


# --- DDP Setup and Model Creation ---
def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def create_base_model(model_name):
    print(f"Creating TimmUnet model with backbone: {model_name}")
    return TimmUnet(backbone_name=model_name, pretrained=True)


# --- Helper function to download snapshot from a previous run ---
def download_snapshot_from_previous_run(experiment_name, output_dir):
    """
    Finds the most recent completed or failed run in the same experiment
    and downloads its training snapshot if one exists. This is robust
    for preempted spot instances.
    """
    try:
        ml_client = MLClient.from_config(credential=DefaultAzureCredential(exclude_shared_cache_credential=True))
        current_run_id = os.environ.get("AZUREML_RUN_ID")

        if not current_run_id:
            print("--> WARNING: AZUREML_RUN_ID not found. Cannot search for previous runs.")
            return None

        print(f"--> Current run ID: {current_run_id}. Searching for previous snapshots in experiment '{experiment_name}'...")

        # List all jobs in the experiment
        all_jobs = list(ml_client.jobs.list(experiment_name=experiment_name))
        
        if not all_jobs:
            print("--> No previous jobs found in this experiment.")
            return None

        # Sort jobs by creation time, newest first
        all_jobs.sort(key=lambda j: j.creation_context.created_at, reverse=True)

        snapshot_filename = f"{experiment_name}_snapshot.pt"
        
        for job in all_jobs:
            # Skip the current run itself
            if job.name == current_run_id:
                continue

            # Only try to restore from terminal states to ensure snapshot is complete
            if job.status not in ["Completed", "Failed", "Canceled"]:
                print(f"--> Skipping run '{job.name}' (status: {job.status}, not terminal)...")
                continue

            print(f"--> Checking previous run '{job.name}' (status: {job.status}) for a snapshot...")
            
            temp_download_dir = Path(output_dir) / f"temp_download_{job.name}"
            
            try:
                # Download all outputs of the previous job to a temporary directory
                ml_client.jobs.download(name=job.name, download_path=str(temp_download_dir), all_outputs=True)
                
                # Search for the snapshot file within the downloaded artifacts
                found_files = list(temp_download_dir.glob(f"**/{snapshot_filename}"))
                
                if found_files:
                    source_snapshot_path = found_files[0]
                    # Create a stable location for the snapshot and copy it there
                    stable_snapshot_dir = Path(output_dir) / "resumed_snapshot"
                    stable_snapshot_dir.mkdir(parents=True, exist_ok=True)
                    destination_snapshot_path = stable_snapshot_dir / snapshot_filename
                    shutil.copy(source_snapshot_path, destination_snapshot_path)
                    
                    print(f"--> SUCCESS: Snapshot found in run '{job.name}' and copied to '{destination_snapshot_path}'.")
                    return str(destination_snapshot_path)
                else:
                    print(f"--> INFO: No snapshot file named '{snapshot_filename}' found in outputs of run '{job.name}'.")
            
            except Exception as e:
                print(f"--> WARNING: Could not download or find snapshot from run '{job.name}'. Error: {e}")
            
            finally:
                # Clean up the temporary directory for this job
                if temp_download_dir.exists():
                    shutil.rmtree(temp_download_dir)

        print("--> No suitable snapshot found in any previous runs.")
        return None

    except Exception as e:
        print(f"--> FATAL: An error occurred while setting up snapshot download. Training will start from scratch. Error: {e}")
        return None


# --- Trainer Class ---
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module, train_data: DataLoader, val_data: DataLoader,
        optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
        gpu_id: int, experiment_name: str, output_dir: str, early_stopper: EarlyStopping,
        snapshot_to_load: str = None
    ):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data, self.val_data = train_data, val_data
        self.optimizer, self.scheduler = optimizer, scheduler
        self.epochs_run = 0
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, "models")
        self.snapshot_dir = os.path.join(output_dir, "snapshots")
        self.snapshot_path = os.path.join(self.snapshot_dir, f"{self.experiment_name}_snapshot.pt")
        self.criterion = nn.MSELoss()
        self.early_stopper = early_stopper
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        if snapshot_to_load and os.path.exists(snapshot_to_load):
            print(f"Loading snapshot from {snapshot_to_load}")
            self._load_snapshot(snapshot_to_load)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"] + 1
        self.early_stopper.best_loss = snapshot.get("BEST_VAL_LOSS", float('inf'))
        self.early_stopper.counter = snapshot.get("PATIENCE_COUNTER", 0)
        print(f"✅ Resuming training from epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        if self.gpu_id != 0: return
        os.makedirs(self.snapshot_dir, exist_ok=True)
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict(),
            "EPOCHS_RUN": epoch,
            "BEST_VAL_LOSS": self.early_stopper.best_loss,
            "PATIENCE_COUNTER": self.early_stopper.counter,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved.")

    def _save_best_model(self):
        if self.gpu_id == 0:
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, f"{self.experiment_name}_best.pth")
            torch.save(self.model.module.state_dict(), model_path)
            print(f"✅ Best model saved at {model_path}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output_norm = self.model(source)
        loss = self.criterion(output_norm, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for source, targets in self.train_data:
            source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
            epoch_loss += self._run_batch(source, targets)
        return epoch_loss / len(self.train_data)

    def _run_validation_epoch(self):
        self.model.eval()
        total_mse, total_mae_physical = 0.0, 0.0
        mae_criterion = nn.L1Loss()
        label_min, label_max = 1500, 4500
        with torch.no_grad():
            for source, targets in self.val_data:
                source, targets = source.to(self.gpu_id), targets.to(self.gpu_id)
                output_norm = self.model(source)
                total_mse += self.criterion(output_norm, targets).item()
                output_physical = output_norm * (label_max - label_min) + label_min
                targets_physical = targets * (label_max - label_min) + label_min
                total_mae_physical += mae_criterion(output_physical, targets_physical).item()
        self.model.train()
        return total_mse / len(self.val_data), total_mae_physical / len(self.val_data)

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            train_loss_mse = self._run_epoch(epoch)
            val_loss_mse, val_loss_mae = self._run_validation_epoch()
            self.scheduler.step()
            
            if self.gpu_id == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} - Train(MSE): {train_loss_mse:.4f} | Val(MSE): {val_loss_mse:.4f} | Val(MAE): {val_loss_mae:.4f} | LR: {current_lr:.6f}")
                
                try:
                    # When running inside an Azure ML job, MLflow is already configured.
                    # We should log to the active run, not create a new nested one.
                    mlflow.log_metric("train_loss_mse", train_loss_mse, step=epoch)
                    mlflow.log_metric("validation_loss_mse", val_loss_mse, step=epoch)
                    mlflow.log_metric("validation_loss_mae", val_loss_mae, step=epoch)
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)
                    print("--> Successfully logged metrics to Azure ML.")
                except Exception as e:
                    print(f"--> WARNING: Could not log metrics to MLflow: {e}")
                
                if self.early_stopper(val_loss_mse):
                    print(f"🏆 New best validation MSE: {self.early_stopper.best_loss:.4f}. Saving best model...")
                    self._save_best_model()
                
                self._save_snapshot(epoch)
                
                if self.early_stopper.early_stop:
                    break
        
        print("✅ Training finished.")

def main_worker(rank, world_size, args, train_dataset, val_dataset, snapshot_path):
    ddp_setup(rank, world_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=DistributedSampler(train_dataset), num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=DistributedSampler(val_dataset), num_workers=0)

    model = create_base_model(args.model_name).to(rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        min_improvement_ratio=args.early_stopping_min_delta
    )

    trainer = Trainer(
        model=model, train_data=train_loader, val_data=val_loader,
        optimizer=optimizer, scheduler=scheduler, gpu_id=rank,
        experiment_name=args.experiment_name, output_dir=args.output_dir,
        early_stopper=early_stopper,
        snapshot_to_load=snapshot_path
    )
    trainer.train(args.num_epochs)
    destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description="PyTorch FWI Training with DDP")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--model_idx", type=int, required=True)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save snapshots and models.")
    
    args = parser.parse_args()
    
    if args.test_mode:
        args.num_epochs = 3 

    snapshot_to_load = download_snapshot_from_previous_run(args.experiment_name, args.output_dir)
    
    splits_file = Path(args.split_dir) / 'geo_aware_splits.json'
    if not splits_file.exists():
        raise FileNotFoundError(f"FATAL: Splits file not found at: {splits_file}")
    
    train_files, val_files, _ = load_specific_split(split_dir=args.split_dir, model_idx=args.model_idx)
    
    train_dataset = SeismicDatasetFileLevel(train_files, args.data_dir, is_train=True)
    val_dataset = SeismicDatasetFileLevel(val_files, args.data_dir, is_train=False)
    
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("ERROR: No CUDA devices found.")
        return

    mp.spawn(main_worker, args=(world_size, args, train_dataset, val_dataset, snapshot_to_load), nprocs=world_size)

if __name__ == '__main__':
    main()