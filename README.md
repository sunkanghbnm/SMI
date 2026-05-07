@ -1,2 +1,97 @@
# SMI
 Hyperspectral Band Selection Benchmark with SMI. A high-performance, parallelized benchmark framework for hyperspectral image (HSI) band selection, featuring the Spectral Markov Strength Index (SMI) based subspace partitioning method.
 "Markov Prior-Based Uniform Subspace Partitioning: An Universal Enhancement Framework for Hyperspectral Band Selection"

## Citation
If you use this code in your research, please cite our paper:
@article{your-paper-citation,
  title={Markov Prior-Based Uniform Subspace Partitioning: An Universal Enhancement Framework for Hyperspectral Band Selection},
  author={Kang Sun et.al},
  journal={XXX},
  year={2026},
  publisher={XXX}
}

## Overview
This repository provides a complete end-to-end pipeline for hyperspectral band selection research:
- Precomputes and caches all statistical similarity matrices (SSIM, covariance, correlation, MIC, etc.)
- Implements classic band selection algorithms and their SMI-enhanced variants
- Supports multiple subspace partitioning strategies (OCF, ASPS, GPC, WALUMI)
- Automates classification performance evaluation with 10-fold independent runs and paired t-test significance analysis
- Fully parallelized using Joblib for multi-core CPU acceleration


## Key Features
Comprehensive Algorithm Library: band selection algorithms including MVPCA, ECA, MCD, Self-Representation, and Similarity Ranking
SMSI Enhancement: Novel spectral Markov property based subspace partitioning for improved band selection
Smart Caching: Automatically caches precomputed statistical matrices to avoid redundant calculations
Standardized Evaluation: LinearSVC classifier with 10 independent runs, paired t-test for statistical significance
Multi-Dataset Support: Works with all standard hyperspectral datasets (Indian Pines, Pavia University, Salinas, Botswana, KSC)
Full Parallelization: All computationally intensive steps are parallelized using Joblib
Reproducible Results: Fixed random seeds and deterministic execution for academic research

## Project Structure
├── smsi.py # Core SMSI calculation and statistical utility functions
├── run_test.py # Main entry point: full pipeline execution
├── data_loader.py # HSI dataset loading utilities
├── config.py # Global configuration (dataset paths, parameters)
├── algorithms/ # Implementation of all band selection algorithms
│ ├── basic_utils.py # Basic statistical and similarity functions
│ ├── mvpca_bs.py # MVPCA algorithm
│ ├── eca_bs.py # ECA algorithm
│ ├── max_covmatrix_det_bs.py # MCD algorithm
│ ├── self_representation_bs.py # Self-representation based algorithm
│ ├── similarity_ranking_bs.py # Similarity ranking based algorithm
│ ├── uniform_sample_bs.py # Uniform sampling baselines
│ └── subspace_partition.py # Subspace partitioning methods
├── datasets/ # Directory for storing .mat format HSI datasets
└── results/ # Auto-generated output directory

## Requirements
- Python 3.10+
- Multi-core CPU (recommended 8+ cores for faster execution)
- 8GB+ RAM (16GB+ recommended for large datasets)

## Quick Start
1. Configure Datasets
Place your hyperspectral datasets in .mat format in the datasets/ directory
2. Update config.py to add your dataset paths:

# Example config.py
DATASET_PATHS = {
    "indianpines": ("indianpines.mat", "indianpines_gt.mat"),
    "pavia": ("Pavia.mat", "Pavia_gt.mat"),
    "salinas": ("Salinas.mat", "Salinas_gt.mat")
}
BASE_DATA_DIR = "./datasets"
OUTPUT_DIR = "./results"

3. Run the Full Pipeline
The main entry point is run_test.py, which executes the complete pipeline:
    Load and preprocess dataset
    Compute and cache all statistical matrices
    Run all band selection algorithms
    Evaluate classification performance with statistical significance analysis
4. Control Execution Steps
In run_test.py, use these three boolean flags to control which parts of the pipeline run:
# In process_single_dataset() function
    is_run_smi = True   # Compute statistical metrics and SMSI
    is_run_bs = True    # Run all band selection algorithms
    is_run_eva = True   # Evaluate classification performance
    Note: Set flags to False to skip completed steps and save time. Precomputed results are automatically loaded from cache.
5. Output Results
All results are automatically saved in the ./results/{dataset_name}/ directory:
    Cached statistical matrices: .npy files for all precomputed similarity matrices
    Band selection results: .npy files containing selected band indices for each algorithm
    Classification results:
    {dataset}_classification_results_raw.csv: Raw accuracy values for all 10 runs
    {dataset}_classification_results_summary.csv: Statistical summary (mean, std, p-values vs baselines)
    Statistical Significance: Results marked with * indicate p < 0.05, ** indicate p < 0.01 (two-sided paired t-test vs corresponding baseline algorithm)
## Supported Band Selection Algorithms

    MVPCA	MVPCA, MVPCA-SMI, MVPCA-OCF, MVPCA-ASPS, MVPCA-GPC, MVPCA-WALUMI
    ECA	ECA, ECA-SMI, ECA-OCF, ECA-ASPS, ECA-GPC, ECA-WALUMI
    MCD	MCD, MCD-SMI, MCD-OCF, MCD-ASPS, MCD-GPC, MCD-WALUMI
    Self-Representation	SEFREP, SEFREP-SMI, SEFREP-OCF, SEFREP-ASPS, SEFREP-GPC, SEFREP-WALUMI
    Similarity Ranking	SIMRNK, SIMRNK-SMI, SIMRNK-OCF, SIMRNK-ASPS, SIMRNK-GPC, SIMRNK-WALUMI
    Baselines	Uniform Sampling, Random Sampling
