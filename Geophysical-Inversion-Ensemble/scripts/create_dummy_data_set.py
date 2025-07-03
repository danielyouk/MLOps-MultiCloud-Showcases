import os
import numpy as np
import json
from data_utils import create_model_splits, get_model_splits

def create_dummy_dataset(output_dir, num_samples=10):
    """
    Create a small dummy dataset for testing.
    
    Args:
        output_dir (str): Directory to save the dummy dataset
        num_samples (int): Number of samples to create per geological type
    """
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create two geological types
    geo_types = ['FlatVel_A', 'FlatFault_A']
    
    for geo_type in geo_types:
        # Create data and model directories
        data_dir = os.path.join(output_dir, geo_type, 'data')
        model_dir = os.path.join(output_dir, geo_type, 'model')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create dummy data files
        for i in range(num_samples):
            # Create dummy seismic data (5, 1000, 70)
            seismic_data = np.random.randn(5, 1000, 70).astype(np.float32)
            np.save(os.path.join(data_dir, f'data{i+1}.npy'), seismic_data)
            
            # Create dummy velocity model (70, 70)
            velocity_model = np.random.randn(70, 70).astype(np.float32)
            np.save(os.path.join(model_dir, f'model{i+1}.npy'), velocity_model)
    
    print(f"\nCreated dummy dataset in {output_dir}")
    print(f"  - {num_samples} samples per geological type")
    print(f"  - Total samples: {num_samples * len(geo_types)}")
    print(f"  - Geological types: {', '.join(geo_types)}")
    
    return output_dir

def main():
    # Create a small dummy dataset
    dummy_dir = "dummy_dataset"
    print(f"Creating dummy dataset in {dummy_dir}...")
    create_dummy_dataset(dummy_dir, num_samples=5)
    
if __name__ == "__main__":
    main() 