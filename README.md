# A Machine Learning Framework for Automated Computational Ethology Using Markerless Pose Estimation

This repository contains the code and experiments corresponding to the publication:

**A Machine Learning Framework for Automated Computational Ethology Using Markerless Pose Estimation**

Publication and DOI link will be provided when available.

## Overview

This work presents an end-to-end framework for automated behavioral classification that bridges SLEAP markerless pose estimation with machine learning approaches. The framework is demonstrated on Drosophila larvae, enabling simultaneous classification of three behavioral states: feeding, sleeping, and crawling.

**Key Features:**

- Integration of SLEAP pose estimation with systematic feature engineering
- Evaluation of 5 ML algorithms across 3 validation strategies
- Position-invariant feature engineering from pose coordinates
- Comprehensive validation addressing temporal dependencies and cross-individual generalization

## Repository Structure

```
├── experiments.ipynb          # Main experimental pipeline and analysis
├── utils.py                   # Core functions for data processing and ML
├── data/                      # SLEAP pose estimation files
│   └── pub_videos_1-10 (inference labels).slp
├── results/                   # Generated outputs and performance metrics
│   ├── *.csv                 # Performance results and summaries
│   ├── *.png                 # Figures and visualizations
│   └── *.pkl                 # Saved confusion matrices
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Dataset characteristics

- **Videos**: 10 labeled videos with 13 Drosophila larvae (not included in the repo due to size)
- **Frames**: 16,552 total frames across 3 behavioral states
- **Behaviors**: Sleeping (23.1%), Feeding (55.4%), Crawling (21.6%)
- **Annotations**: Expert-labeled behaviors (dictionary available in the notebook file) and SLEAP pose landmarks (4 anatomical points) (available in the `.slp` file)

## Experimental Design

### Feature Engineering (12 features total)

- **6 features**: Center-normalized coordinates (position-invariant)
- **4 features**: Inter-landmark distances (postural signatures)
- **1 feature**: Body curvature (head-center-tail angle)
- **1 feature**: Activity level (velocity-based movement measure)

### Machine Learning Models

- Random Forest (50 trees)
- Support Vector Machine (RBF kernel)
- Gradient Boosting (50 stages)
- K-Nearest Neighbors (k=5)
- Neural Network (12→6→3 architecture)

### Validation Strategies

1. **Stratified Random Split**: Maintains class balance but may inflate performance due to temporal correlations
2. **Temporal Block Split**: 15-frame chunks ensuring temporal independence while preserving behavioral context
3. **Animal-Level Split**: Complete animals as test subjects for cross-individual generalization assessment

## Usage

### Prerequisites

Please install SLEAP using the instructions at https://sleap.ai/develop/installation.html, preferably in a dedicated conda environment.

### Running Experiments

1. **Open the main notebook**: `experiments.ipynb`
2. **Execute cells sequentially** to reproduce all experiments:

   - Data loading and cleaning
   - Feature engineering
   - Machine learning experiments across all validation strategies
   - Visualization generation

3. **Key outputs** are automatically saved to `results/` directory:
   - Performance metrics (CSV files)
   - Behavior timeline visualizations (PNG)
   - Confusion matrices (PNG and PKL)

## Citation

If you use this code or methodology in your research, please cite:\\
`citation pending publication`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues regarding this code, please contact:

- Prateek Verma: prateek@uark.edu
