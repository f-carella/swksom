# swksom: Solar Wind Classification with Self-Organizing Maps and K-means Clustering

Classify solar wind observations into different wind types (CSW, Ejecta, HAW, SSW) using a pre-trained Self-Organizing Map (SOM) model trained on OMNI HRO2 reference data.

## Quick Start

**See [example.ipynb](example.ipynb) for a complete walkthrough.**

The notebook demonstrates the full workflow:
1. **Load Pre-trained Model**: Import the trained toroidal SOM and preprocessing transforms
2. **Preprocess Data**: Apply Box-Cox and log transforms, then scale with RobustScaler
3. **Predict Clusters**: Classify solar wind data into one of four wind types
4. **Visualize Results**: Create multi-variate time series plots color-coded by wind type

## Installation

```bash
conda env create -f swsom.yaml
conda activate swsom
```

Or manually install dependencies:
```bash
pip install numpy pandas scikit-learn scipy matplotlib kneed psutil
```

## Project Contents

| File/Folder | Purpose |
|-------------|---------|
| `example.ipynb` | Complete tutorial showing data preprocessing and classification |
| `ClusterSOM.py` | Main SOM + K-means clustering implementation |
| `cluster_som_utils.py` | Utility functions for clustering and evaluation |
| `models/toroidal_som.pkl` | Pre-trained SOM model |
| `transform/` | Box-Cox and RobustScaler fit objects for preprocessing |
| `data/` | Example solar wind datasets (minimum, maximum, storm events, etc.) |
| `swsom.yaml` | Conda environment specification |

## Input Features

The model expects 8 solar wind features (preprocessed):
- **V** (Bulk Speed) → Box-Cox transformed
- **Np** (Proton Density) → log-scaled
- **B** (Magnetic Field) → log-scaled
- **Tp** (Proton Temperature) → log-scaled
- **Texp/Tp** (Temperature Ratio) → log-scaled
- **Pdyn** (Dynamic Pressure) → log-scaled
- **logBeta** (Plasma Beta)
- **B_rms** (Field Std Dev) → log-scaled

## Wind Types

The model classifies solar wind into four categories:
- **CSW**: Compressed Solar Wind
- **Ejecta**: Ejecta (ICME's Magnetic Obstacle)
- **HAW**: Highly-Alfvénic Wind
- **SSW**: Slow Solar Wind
