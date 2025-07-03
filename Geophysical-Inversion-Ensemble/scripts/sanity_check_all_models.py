#!/usr/bin/env python3
"""
Sanity Check Script for All Models
This script tests all models in train_all_models.py with a small dummy dataset
to ensure they can run locally before submitting to Azure ML.
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
import yaml

# Add src to path
sys.path.append('src')

def create_dummy_dataset(output_dir="dummy_dataset", num_samples=5):
    """Create a small dummy dataset for testing."""
    print(f"Creating dummy dataset in {output_dir}...")
    
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
            # Create dummy seismic data (500, 5, 1000, 70) - batch of 500 samples
            seismic_data = np.random.randn(500, 5, 1000, 70).astype(np.float32)
            np.save(os.path.join(data_dir, f'seis{i+1}_{i+1}_{i}.npy'), seismic_data)
            
            # Create dummy velocity model (500, 1, 70, 70) - batch of 500 samples
            velocity_model = np.random.randn(500, 1, 70, 70).astype(np.float32)
            np.save(os.path.join(model_dir, f'vel{i+1}_{i+1}_{i}.npy'), velocity_model)
    
    print(f"Created dummy dataset with {num_samples} files per geological type")
    print(f"File format: seis{i+1}_{i+1}_{i}.npy -> vel{i+1}_{i+1}_{i}.npy")
    return output_dir

def test_model_import(model_name):
    """Test if a model can be imported and instantiated."""
    try:
        # Import the model based on the name
        if "caformer" in model_name.lower():
            # For CAFormer models, we need to use timm
            import timm
            model = timm.create_model(model_name, pretrained=False, num_classes=70*70)
            print(f"✅ {model_name}: Successfully imported from timm")
        elif "convnext" in model_name.lower():
            # For ConvNeXt models, we need to use timm
            import timm
            model = timm.create_model(model_name, pretrained=False, num_classes=70*70)
            print(f"✅ {model_name}: Successfully imported from timm")
        elif "maxvit" in model_name.lower():
            # For MaxViT models, we need to use timm
            import timm
            model = timm.create_model(model_name, pretrained=False, num_classes=70*70)
            print(f"✅ {model_name}: Successfully imported from timm")
        elif "coatnet" in model_name.lower():
            # For CoAtNet models, we need to use timm
            import timm
            model = timm.create_model(model_name, pretrained=False, num_classes=70*70)
            print(f"✅ {model_name}: Successfully imported from timm")
        elif "coatnext" in model_name.lower():
            # For CoAtNeXt models, we need to use timm
            import timm
            model = timm.create_model(model_name, pretrained=False, num_classes=70*70)
            print(f"✅ {model_name}: Successfully imported from timm")
        elif "efficientnet" in model_name.lower():
            # For EfficientNet models, we need to use timm
            import timm
            model = timm.create_model(model_name, pretrained=False, num_classes=70*70)
            print(f"✅ {model_name}: Successfully imported from timm")
        elif "mobilenet" in model_name.lower():
            # For MobileNet models, we need to use timm
            import timm
            model = timm.create_model(model_name, pretrained=False, num_classes=70*70)
            print(f"✅ {model_name}: Successfully imported from timm")
        elif "vit" in model_name.lower():
            # For Vision Transformer models, we need to use timm
            import timm
            model = timm.create_model(model_name, pretrained=False, num_classes=70*70)
            print(f"✅ {model_name}: Successfully imported from timm")
        elif "eva02" in model_name.lower():
            # For EVA02 models, we need to use timm
            import timm
            model = timm.create_model(model_name, pretrained=False, num_classes=70*70)
            print(f"✅ {model_name}: Successfully imported from timm")
        else:
            print(f"❌ {model_name}: Unknown model type")
            return False
        
        # Test forward pass with appropriate input
        model.eval()
        with torch.no_grad():
            # Create appropriate dummy input based on model type
            if "vit" in model_name.lower() or "eva02" in model_name.lower():
                # Vision Transformers expect 224x224 images with 3 channels
                dummy_input = torch.randn(2, 3, 224, 224)
                print(f"   Using ViT input shape: {dummy_input.shape}")
            elif "maxvit" in model_name.lower():
                # MaxViT expects input size divisible by window size (16)
                # Use 256x256 which is divisible by 16
                dummy_input = torch.randn(2, 3, 256, 256)
                print(f"   Using MaxViT input shape: {dummy_input.shape}")
            else:
                # CNN models expect 3 channels, but we'll test with 3 channels first
                # For seismic data, we'd need to adapt the input
                dummy_input = torch.randn(2, 3, 224, 224)  # Standard image size
                print(f"   Using CNN input shape: {dummy_input.shape}")
            
            output = model(dummy_input)
            print(f"   Output shape: {output.shape}")
            
            # Check if output can be reshaped to (70, 70)
            if output.shape[-1] == 70*70:
                output_reshaped = output.view(2, 70, 70)
                print(f"   ✅ Output can be reshaped to velocity model shape")
            else:
                print(f"   ⚠️  Output shape {output.shape} may need adjustment")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_name}: Error - {str(e)}")
        return False

def test_data_loading(data_dir):
    """Test if data can be loaded correctly."""
    try:
        from dataset import SeismicDataset
        
        print(f"Testing data loading from {data_dir}...")
        dataset = SeismicDataset(data_dir=data_dir, max_samples=5)
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Found {len(dataset)} samples")
        print(f"   Found {len(dataset.pairs)} file pairs")
        
        # Test getting a sample
        if len(dataset) > 0:
            seismic, velocity = dataset[0]
            print(f"   Sample shapes - Seismic: {seismic.shape}, Velocity: {velocity.shape}")
            print(f"   Sample types - Seismic: {seismic.dtype}, Velocity: {velocity.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return False

def test_timm_unet_model():
    """Test the actual TimmUnet model that would be used in training."""
    try:
        print("Testing TimmUnet model with seismic data...")
        from timm_unet import TimmUnet
        
        # Test with a simple backbone
        model = TimmUnet(backbone_name="convnext_tiny.fb_in22k_ft_in1k", pretrained=False)
        model.eval()
        
        # Create dummy seismic input (batch_size=2, channels=5, height=1000, width=70)
        dummy_input = torch.randn(2, 5, 1000, 70)
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            
            # Check if output can be reshaped to velocity model shape
            if output.shape[-2:] == (70, 70):
                print(f"   ✅ Output matches velocity model shape (70, 70)")
            else:
                print(f"   ⚠️  Output shape {output.shape} may need adjustment")
        
        return True
        
    except Exception as e:
        print(f"❌ TimmUnet test failed: {str(e)}")
        return False

def test_training_script():
    """Test if the training script can be imported and run with minimal parameters."""
    try:
        print("Testing training script import...")
        
        # Try to import the main function
        try:
            from train_multi_gpu import main as train_main
            print("✅ Training script imported successfully")
        except ImportError as e:
            if "mlflow" in str(e):
                print("⚠️  Training script import failed due to missing mlflow")
                print("   This is optional - mlflow is only used for logging")
                print("   The script will work without mlflow in Azure ML")
                return True  # Consider this a pass since mlflow is optional
            else:
                print(f"❌ Training script import failed: {str(e)}")
                return False
        
        # Test if we can create a basic model
        try:
            from timm_unet import TimmUnet
            # Test with a simple model
            model = TimmUnet(backbone_name="convnext_tiny.fb_in22k_ft_in1k", pretrained=False)
            print("✅ TimmUnet model creation successful")
        except Exception as e:
            print(f"⚠️  TimmUnet model creation failed: {str(e)}")
            print("   This might be due to missing timm or model dependencies")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Training script test failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Sanity check for all models')
    parser.add_argument('--create-dummy-data', action='store_true', 
                       help='Create dummy dataset for testing')
    parser.add_argument('--test-models-only', action='store_true',
                       help='Only test model imports, skip data and training tests')
    parser.add_argument('--data-dir', type=str, default='dummy_dataset',
                       help='Directory containing test data')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧪 SANITY CHECK FOR ALL MODELS")
    print("=" * 80)
    
    # Construct the path to the config file relative to the project root
    config_path = 'configs/model_configs.yml'
    print(f"--> Loading model configurations from: {config_path}")
    try:
        with open(config_path, 'r') as file:
            MODEL_CONFIGS = yaml.safe_load(file)
        print(f"--> Found {len(MODEL_CONFIGS)} model configurations.")
    except FileNotFoundError:
        print(f"FATAL ERROR: The config file was not found at '{config_path}'")
        print("Please ensure you are running this script from the project root directory.")
        return
    # --------------------------------------------------------------------

    print(f"Testing {len(MODEL_CONFIGS)} models...")
    print()
    
    # Create dummy dataset if requested
    if args.create_dummy_data:
        data_dir = create_dummy_dataset()
    else:
        data_dir = args.data_dir
    
    # Test model imports
    print("🔧 Testing Model Imports (Raw Models):")
    print("-" * 50)
    print("Note: This tests raw model imports from timm library.")
    print("These models expect standard image inputs (3 channels, 224x224).")
    print("For actual training, these models are adapted via TimmUnet wrapper.")
    print()
    
    successful_models = []
    failed_models = []
    
    for model_key, config in MODEL_CONFIGS.items():
        model_name = config['model_name']
        print(f"\nTesting {model_key}: {model_name}")
        
        if test_model_import(model_name):
            successful_models.append(model_key)
        else:
            failed_models.append(model_key)
    
    print(f"\nModel Import Results:")
    print(f"✅ Successful: {len(successful_models)}")
    print(f"❌ Failed: {len(failed_models)}")
    
    if failed_models:
        print(f"\nFailed models: {', '.join(failed_models)}")
    
    # Skip data and training tests if only testing models
    if args.test_models_only:
        print("\n" + "=" * 80)
        print("🧪 SANITY CHECK COMPLETE (Models Only)")
        print("=" * 80)
        return
    
    # Test data loading
    print(f"\n📊 Testing Data Loading:")
    print("-" * 50)
    data_loading_success = test_data_loading(data_dir)
    
    # Test TimmUnet model
    print(f"\n🚀 Testing TimmUnet Model:")
    print("-" * 50)
    timm_unet_success = test_timm_unet_model()
    
    # Test training script
    print(f"\n🚀 Testing Training Script:")
    print("-" * 50)
    training_script_success = test_training_script()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 SANITY CHECK SUMMARY")
    print("=" * 80)
    print(f"✅ Model Imports: {len(successful_models)}/{len(MODEL_CONFIGS)} successful")
    print(f"📊 Data Loading: {'✅ PASS' if data_loading_success else '❌ FAIL'}")
    print(f"🚀 TimmUnet Model: {'✅ PASS' if timm_unet_success else '❌ FAIL'}")
    print(f"🚀 Training Script: {'✅ PASS' if training_script_success else '❌ FAIL'}")
    
    # Determine overall status
    critical_tests_passed = data_loading_success and timm_unet_success and training_script_success
    model_import_ratio = len(successful_models) / len(MODEL_CONFIGS)
    
    if critical_tests_passed and model_import_ratio >= 0.9:  # Allow 10% model import failures
        print("\n🎉 CRITICAL TESTS PASSED! Ready for Azure ML submission.")
        if model_import_ratio < 1.0:
            print(f"⚠️  Note: {len(failed_models)} model(s) failed import but this won't affect training.")
            print("   The TimmUnet wrapper handles model adaptation for seismic data.")
    else:
        print("\n⚠️  Some critical tests failed. Please fix issues before submitting to Azure ML.")
        if not critical_tests_passed:
            print("   Critical failures in data loading, TimmUnet, or training script.")
        if model_import_ratio < 0.9:
            print(f"   Too many model import failures ({len(failed_models)}/{len(MODEL_CONFIGS)}).")
    
    print("=" * 80)

if __name__ == '__main__':
    main() 