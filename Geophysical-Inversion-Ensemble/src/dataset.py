import os
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import re

class SeismicDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the seismic and velocity map data.
    This dataset handles .npy files that contain a batch of samples.
    It maps a global index to a specific file and a specific slice within that file.
    
    Supports two folder structures:
    1. Direct files (Fault types): seis*.npy and vel*.npy directly in geo_type directory
    2. Subdirectory structure (Velocity/Style types): data/data*.npy and model/model*.npy
    """
    def __init__(self, data_dir, transform=None, pair_type="data_model", max_samples=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Path to the data directory
            transform (callable, optional): Optional transform to be applied on a sample
            pair_type (str): Type of data pairs to use ('data_model', 'seis_vel', or 'all')
            max_samples (int, optional): Maximum number of samples to use (for testing)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.pair_type = pair_type
        self.max_samples = max_samples
        self.label_min = 1500
        self.label_max = 4500
        self.pairs = []
        self.sample_indices = []  # Store (file_pair_idx, sample_idx) for each global index
        print(f"DEBUG: Starting recursive search in {data_dir}")
        
        # For each geological type directory
        for geo_type in os.listdir(data_dir):
            geo_path = os.path.join(data_dir, geo_type)
            if not os.path.isdir(geo_path):
                continue
                
            print(f"Processing geological type: {geo_type}")
            
            # Check if this is a direct file structure (Fault types)
            seis_files = sorted([f for f in os.listdir(geo_path) if f.startswith('seis') and f.endswith('.npy')])
            vel_files = sorted([f for f in os.listdir(geo_path) if f.startswith('vel') and f.endswith('.npy')])
            
            if seis_files and vel_files:
                # Direct file structure (Fault types)
                print(f"  Found direct file structure with {len(seis_files)} seismic and {len(vel_files)} velocity files")
                
                # Pair seismic and velocity files by their base name
                for seis_file in seis_files:
                    # Extract the base name (e.g., '4_1_0' from 'seis4_1_0.npy')
                    base_name = seis_file.replace('seis', '').replace('.npy', '')
                    vel_file = f"vel{base_name}.npy"
                    
                    if vel_file in vel_files:
                        seis_path = os.path.join(geo_path, seis_file)
                        vel_path = os.path.join(geo_path, vel_file)
                        self.pairs.append((seis_path, vel_path))
                        print(f"    Paired: {seis_file} -> {vel_file}")
            
            else:
                # Check for subdirectory structure (Velocity/Style types)
                data_subdir = os.path.join(geo_path, 'data')
                model_subdir = os.path.join(geo_path, 'model')
                data_files = []
                model_files = []
                if os.path.isdir(data_subdir) and os.path.isdir(model_subdir):
                    print(f"  Found subdirectory structure")
                    data_files = sorted([f for f in os.listdir(data_subdir) if f.startswith('data') and f.endswith('.npy')])
                    model_files = sorted([f for f in os.listdir(model_subdir) if f.startswith('model') and f.endswith('.npy')])
                    print(f"    Found {len(data_files)} data files and {len(model_files)} model files")
                # Pair data and model files by their number
                for data_file in data_files:
                    # Extract the number (e.g., '1' from 'data1.npy')
                    data_num = data_file.replace('data', '').replace('.npy', '')
                    model_file = f"model{data_num}.npy"
                    if model_file in model_files:
                        data_path = os.path.join(data_subdir, data_file)
                        model_path = os.path.join(model_subdir, model_file)
                        self.pairs.append((data_path, model_path))
                        print(f"    Paired: {data_file} -> {model_file}")
        
        print(f"Found {len(self.pairs)} total file pairs.")
        
        # Create sample indices for each file pair
        # Each .npy file contains 500 samples
        SAMPLES_PER_FILE = 500
        for file_pair_idx in range(len(self.pairs)):
            for sample_idx in range(SAMPLES_PER_FILE):
                self.sample_indices.append((file_pair_idx, sample_idx))
        
        print(f"Total samples available: {len(self.sample_indices)}")
        
        # Apply max_samples limit if specified
        if max_samples is not None:
            self.sample_indices = self.sample_indices[:max_samples]
            print(f"Using a subset of {len(self.sample_indices)} samples for this test run.")
        
        print("First few pairs:")
        for i, (input_file, target_file) in enumerate(self.pairs[:3]):
            print(f"  {i+1}. {os.path.basename(input_file)} -> {os.path.basename(target_file)}")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.sample_indices):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.sample_indices)} samples")
        
        file_pair_idx, sample_idx = self.sample_indices[idx]
        input_file, target_file = self.pairs[file_pair_idx]
        
        # Load data
        seismic_data = np.load(input_file, mmap_mode='r')  # Shape: (500, 5, 1000, 70)
        velocity_model = np.load(target_file, mmap_mode='r')  # Shape: (500, 1, 70, 70)
        
        # Extract the specific sample from the batch
        seismic_sample = seismic_data[sample_idx]  # Shape: (5, 1000, 70)
        velocity_sample = velocity_model[sample_idx]  # Shape: (1, 70, 70)
        
        # Convert to torch tensors
        seismic_tensor = torch.from_numpy(seismic_sample).float()
        velocity_tensor = torch.from_numpy(velocity_sample).float()
        
        # Apply transforms if any
        if self.transform:
            seismic_tensor = self.transform(seismic_tensor)
            velocity_tensor = self.transform(velocity_tensor)
        
        return seismic_tensor, velocity_tensor


class SeismicDatasetWithSplits(Dataset):
    """
    Custom PyTorch Dataset for loading seismic and velocity map data using pre-defined splits.
    This dataset works with sample-level splits created by create_sample_level_splits.
    """
    def __init__(self, split_samples, data_dir, transform=None, max_samples=None):
        """
        Initialize the dataset with pre-defined sample splits.
        
        Args:
            split_samples (list): List of sample dictionaries from load_sample_splits
            data_dir (str): The root directory where the data files are located (mount path in Azure).
            transform (callable, optional): Optional transform to be applied on a sample
            max_samples (int, optional): Maximum number of samples to use (for testing)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples = max_samples
        self.samples = split_samples
        
        # Apply max_samples limit if specified
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
            print(f"Using a subset of {len(self.samples)} samples for this test run.")
        
        print(f"Loaded {len(self.samples)} samples from split")
        
        # Show sample distribution by geological type
        geo_types = {}
        for sample in self.samples:
            geo_type = sample['geo_type']
            geo_types[geo_type] = geo_types.get(geo_type, 0) + 1
        
        print(f"Sample distribution by geological type: {geo_types}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.samples)} samples")
        
        sample_info = self.samples[idx]
        # Use os.path.normpath for OS-independent path
        input_file = os.path.normpath(os.path.join(self.data_dir, sample_info['input_file']))
        target_file = os.path.normpath(os.path.join(self.data_dir, sample_info['target_file']))
        sample_idx = sample_info['sample_idx']
        
        # Load data
        seismic_data = np.load(input_file)  # Shape: (500, 5, 1000, 70)
        velocity_model = np.load(target_file)  # Shape: (500, 1, 70, 70)
        
        # Extract the specific sample from the batch
        seismic_sample = seismic_data[sample_idx]  # Shape: (5, 1000, 70)
        velocity_sample = velocity_model[sample_idx]  # Shape: (1, 70, 70)
        
        # Convert to torch tensors
        seismic_tensor = torch.from_numpy(seismic_sample).float()
        velocity_tensor = torch.from_numpy(velocity_sample).float()
        
        # Apply transforms if any
        if self.transform:
            seismic_tensor = self.transform(seismic_tensor)
            velocity_tensor = self.transform(velocity_tensor)
        
        return seismic_tensor, velocity_tensor


class SeismicDatasetFileLevel(Dataset):
    """
    Custom PyTorch Dataset for loading seismic and velocity map data using file-level splits.
    This version includes on-the-fly data augmentation for the training set.
    """
    def __init__(self, file_list, data_dir, is_train=False, transform=None, max_samples=None):
        """
        Initialize the dataset with file-level splits.
        
        Args:
            file_list (list): List of file dictionaries from file-level splits.
            data_dir (str): The root directory where the data files are located.
            is_train (bool): Flag to enable/disable augmentations. Set to True for the training set.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_samples (int, optional): Maximum number of samples to use (for testing).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples = max_samples
        self.file_list = file_list
        self.is_train = is_train  # <-- ADDED: Flag for augmentation
        
        # Normalization parameters
        self.label_min = 1500
        self.label_max = 4500
        
        SAMPLES_PER_FILE = 500
        self.sample_indices = []
        
        for file_idx, file_info in enumerate(file_list):
            for sample_idx in range(SAMPLES_PER_FILE):
                self.sample_indices.append((file_idx, sample_idx))
        
        if max_samples is not None:
            self.sample_indices = self.sample_indices[:max_samples]
        
        print(f"Loaded {len(self.sample_indices)} samples from {len(file_list)} files. Augmentation enabled: {self.is_train}")

    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        file_idx, sample_idx = self.sample_indices[idx]
        file_info = self.file_list[file_idx]
        
        input_file = os.path.normpath(os.path.join(self.data_dir, file_info['input']))
        target_file = os.path.normpath(os.path.join(self.data_dir, file_info['target']))
        
        seismic_data = np.load(input_file)
        velocity_model = np.load(target_file)
        
        seismic_sample = seismic_data[sample_idx]
        velocity_sample = velocity_model[sample_idx]
        
        # --- THIS IS THE DATA AUGMENTATION LOGIC ---
        # It only runs if is_train=True, inspired by the high-performing notebook
        if self.is_train and np.random.rand() < 0.5:
            # Flip the geophone axis (the last dimension) for both input and target
            seismic_sample = np.ascontiguousarray(np.flip(seismic_sample, axis=-1))
            velocity_sample = np.ascontiguousarray(np.flip(velocity_sample, axis=-1))
        # -----------------------------------------

        seismic_tensor = torch.from_numpy(seismic_sample).float()
        velocity_tensor = torch.from_numpy(velocity_sample).float()
        
        if self.transform:
            seismic_tensor = self.transform(seismic_tensor)
            velocity_tensor = self.transform(velocity_tensor)
        
        # Normalize the target velocity map to a [0, 1] range
        velocity_tensor = (velocity_tensor - self.label_min) / (self.label_max - self.label_min)
        
        return seismic_tensor, velocity_tensor


# This main block is for testing the dataset implementation
if __name__ == '__main__':
    print("--- Testing SeismicDataset ---")
    # This path is relative to the project root (MyKaggleProject)
    test_data_dir = './Yale_UNC-CH_Geophysical_Waveform_Inversion/data/train_samples/'
    
    try:
        dataset = SeismicDataset(data_dir=test_data_dir, max_samples=10)
        print(f"Successfully found {len(dataset.pairs)} file pairs.")
        print(f"Total samples available: {len(dataset)}")
        
        # Get a sample
        print("Fetching a sample from the dataset...")
        seismic_sample, model_sample = dataset[0] # Get the first sample
        
        # Check shapes and types
        print(f"Sample seismic shape: {seismic_sample.shape}, type: {seismic_sample.dtype}")
        print(f"Sample model shape: {model_sample.shape}, type: {model_sample.dtype}")
        
        # Test getting another sample
        print("Fetching another sample...")
        seismic_sample2, model_sample2 = dataset[1]
        print(f"Second sample seismic shape: {seismic_sample2.shape}")

        print("\n--- SeismicDataset test passed! ---")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data directory exists and is populated.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")