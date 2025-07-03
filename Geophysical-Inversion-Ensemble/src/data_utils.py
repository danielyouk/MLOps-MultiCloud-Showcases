# src/data_utils.py

import os
import json
import numpy as np
from pathlib import Path

def create_geo_aware_file_splits(data_dir, split_dir, num_models=16, random_state=42):
    """
    Creates robust, file-level splits stratified by geological type.
    It reserves some geo types for a fixed test set and creates N different
    train/val splits for ensembling. This should be run once before training.
    """
    data_dir = Path(data_dir)
    split_dir = Path(split_dir)
    split_dir.mkdir(exist_ok=True, parents=True)

    geo_type_files = {}
    print("--> Scanning for data files...")

    # Step 1: Find all file pairs and group by geological type
    for geo_path in data_dir.iterdir():
        if not geo_path.is_dir():
            continue
        
        geo_type = geo_path.name
        geo_type_files[geo_type] = []
        
        seis_files_direct = sorted(geo_path.glob("seis*.npy"))
        data_files_subdir = sorted(geo_path.glob("data/data*.npy"))

        if seis_files_direct: # Handles 'seis'/'vel' naming
            for seis_path in seis_files_direct:
                vel_path = geo_path / seis_path.name.replace("seis", "vel")
                if vel_path.exists():
                    file_pair = {'input': str(seis_path.relative_to(data_dir)), 'target': str(vel_path.relative_to(data_dir))}
                    geo_type_files[geo_type].append(file_pair)
        elif data_files_subdir: # Handles 'data'/'model' naming
            for data_path in data_files_subdir:
                model_path = geo_path / "model" / data_path.name.replace("data", "model")
                if model_path.exists():
                    file_pair = {'input': str(data_path.relative_to(data_dir)), 'target': str(model_path.relative_to(data_dir))}
                    geo_type_files[geo_type].append(file_pair)
    
    print(f"--> Found files for {len(geo_type_files)} geological types.")

    # Step 2: Define a fixed hold-out test set
    all_geo_types = sorted(geo_type_files.keys())
    test_geo_types = ['CurveFault_B', 'Style_A', 'FlatVel_B'] 
    train_val_geo_types = [gt for gt in all_geo_types if gt not in test_geo_types]
    
    test_files = [file for gt in test_geo_types for file in geo_type_files.get(gt, [])]
    print(f"\n--> Created fixed hold-out test set with {len(test_files)} files from {len(test_geo_types)} geo types: {test_geo_types}")

    # Step 3: Create N unique train/val splits
    np.random.seed(random_state)
    all_splits = {}
    print("\n--> Generating unique train/validation splits for each model...")
    for i in range(num_models):
        shuffled_types = train_val_geo_types.copy()
        np.random.shuffle(shuffled_types)
        
        val_type = shuffled_types[i % len(shuffled_types)]
        train_types = [gt for gt in shuffled_types if gt != val_type]
        
        train_files = [file for gt in train_types for file in geo_type_files.get(gt, [])]
        val_files = geo_type_files.get(val_type, [])
        
        all_splits[i] = {
            'train': train_files,
            'val': val_files,
            'test': test_files,
            'val_geo_type': val_type
        }

    # Step 4: Save all splits to a single JSON file
    splits_file_path = split_dir / 'geo_aware_splits.json'
    with open(splits_file_path, 'w') as f:
        json.dump(all_splits, f, indent=2)
    print(f"\n--> All {num_models} splits saved successfully to {splits_file_path}")
    return all_splits

def load_specific_split(split_dir, model_idx):
    """
    Loads the train, val, and test file lists for a specific model index
    from the main geo-aware splits file. This is used by the training script.
    """
    splits_file = Path(split_dir) / 'geo_aware_splits.json'
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found at {splits_file}. Please run create_geo_aware_file_splits first.")
    
    with open(splits_file, 'r') as f:
        all_splits = json.load(f)
    
    model_splits = all_splits.get(str(model_idx))
    if not model_splits:
        raise KeyError(f"Model index {model_idx} not found in the splits file.")
        
    print(f"Loaded data split for model_idx: {model_idx}")
    return model_splits['train'], model_splits['val'], model_splits['test']