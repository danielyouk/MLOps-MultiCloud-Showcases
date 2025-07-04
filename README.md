# MLOps-Showcase-MultiCloud
An advanced showcase of end-to-end MLOps pipelines deployed across AWS, Azure, and GCP. This project demonstrates automated, scalable, and parallelized machine learning workflows, from data ingestion to model serving, for a variety of models including LLMs, Neural Networks, and classical ML

# Project Structure
```text
MLOps-Portfolio/
├── .gitignore              # ✅ A single gitignore at the root for all projects

└── Geophysical-Inversion-Ensemble/
    ├── README.md               # Project-specific description, setup, and results
    ├── requirements.txt        # All Python packages needed for this project
    │
    ├── configs/                # For all configuration files
    │   └── model_configs.yml     # Defines the model portfolio and hyperparameters
    │
    ├── data/                   # (This folder should be in .gitignore)
    │   ├── raw/                  # For the original, untouched competition data
    │   │   └── train_samples/
    │   └── splits/               # For the generated data split configurations
    │       └── geo_aware_splits.json
    │
    ├── outputs/                # (This folder should be in .gitignore)
    │   ├── models/               # Where trained model checkpoints (.pth) are saved
    │   ├── logs/                 # For logs from local runs or Azure job downloads
    │   └── holdout_preds/        # For saved predictions from the ensemble notebook
    │
    ├── scripts/                # All executable "runner" scripts
    │   ├── 1_Ensemble_Weight_Finder.ipynb   # For finding optimal weights on Kaggle
    │   ├── 2_Final_Submission_Generator.ipynb # For creating the final submission on Kaggle
    │   ├── check_jobs.py         # To check status and get logs from Azure ML
    │   ├── create_file_splits.py # To generate the geo_aware_splits.json
    │   ├── run_azureml.py        # The script that submits a single job to Azure
    │   └── train_all_models.py   # The main orchestrator you run locally
    │
    ├── src/                    # All core source code (your "library")
    │   ├── __init__.py           # Makes 'src' a Python package
    │   ├── data_utils.py       # Functions for creating and loading data splits
    │   ├── dataset.py          # The PyTorch Dataset class
    │   └── models/               # Sub-package for model architectures
    │       ├── __init__.py
    │       ├── public_nets.py    # A place to put the Net_backbone classes
    │       └── timm_unet.py      # Your universal TimmUnet architecture
    │
    └── tests/                    # For all testing and validation scripts
        ├── __init__.py
        └── sanity_check_all_models.py # Your script to test models locally
```