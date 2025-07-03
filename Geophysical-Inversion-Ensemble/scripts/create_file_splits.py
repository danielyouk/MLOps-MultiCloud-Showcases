# create_file_splits.py

import sys
import argparse
from pathlib import Path

# Add src directory to path to find data_utils
sys.path.append('src')
from data_utils import create_geo_aware_file_splits

def main():
    parser = argparse.ArgumentParser(description='Create Geo-Aware File-Level Data Splits')
    parser.add_argument('--data_dir', type=str, default='data/train_samples',
                        help='Directory containing the training data')
    parser.add_argument('--split_dir', type=str, default='experiments/file_splits',
                        help='Directory to save the final splits JSON file')
    parser.add_argument('--num_models', type=int, default=16,
                        help='Number of models to create splits for')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CREATING GEO-AWARE FILE-LEVEL DATA SPLITS")
    print("=" * 60)
    
    # Create the file-level splits using the new, safer function
    splits = create_geo_aware_file_splits(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        num_models=args.num_models,
        random_state=args.random_state
    )
    
    print("\n" + "=" * 60)
    print("SCRIPT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"A single splits file has been saved to: {Path(args.split_dir) / 'geo_aware_splits.json'}")
    print("\nSplit Summary:")
    for model_idx, model_splits in splits.items():
        print(f"  Model {model_idx}:")
        print(f"    Train files: {len(model_splits['train']):<5} | Val files: {len(model_splits['val']):<5} (Val Geo: {model_splits['val_geo_type']}) | Test files: {len(model_splits['test'])}")
    
    print("\nThis new split file prevents data leakage by ensuring the test set is completely held out.")

if __name__ == '__main__':
    main()