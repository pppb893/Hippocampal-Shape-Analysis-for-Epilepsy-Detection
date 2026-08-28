# Hippocampal Shape Analysis for Epilepsy Detection

This repository contains a comprehensive, end-to-end pipeline for the automated extraction, 3D shape analysis, and machine learning classification of the hippocampus from structural MRI scans. It is specifically designed to aid in the detection and lateralization of Epilepsy (e.g., Temporal Lobe Epilepsy) using advanced morphometric features.

## 🧠 Project Overview

The project is divided into several main components, guiding the data from raw MRI images all the way to statistical analysis and predictive modeling:

### 1. Data Preprocessing (`run_pipeline.py`)
- **Input:** Raw T1-weighted MRI images (`.nii.gz`, `.mgz`).
- **Process:** 
  - Utilizes **FastSurfer** to automatically segment the brain and extract neuroanatomical labels.
  - Extracts the left (Label 17) and right (Label 53) hippocampus using `extract_hippocampus.py`.
- **Output:** Clean, isolated 3D binary masks of the hippocampus.

### 2. Shape Analysis Pipeline (`run_everything.bat`)
This is the core morphological analysis workflow, driven automatically by SlicerSALT:
- **Step 1: Group-wise ICP Alignment (`ICP/`)** 
  - Runs Iterative Closest Point (ICP) to rigidly align all hippocampal volumes in a dataset into a common physical space. This ensures shape comparisons are rotationally and translationally invariant.
- **Step 2: Batch SPHARM Processing (`SPHARM/`)**
  - Parameterizes the 3D surfaces using Spherical Harmonics (SPHARM-PDM). It creates a uniform grid of vertices across all subjects, establishing true point-to-point anatomical correspondence.
- **Step 3: SPHARM Re-alignment**
  - Fine-tunes the alignment of the SPHARM meshes based on anatomical landmarks (e.g., head, tail, medial, lateral axes).
- **Step 4 & 5: Statistical Analysis (`Visualize/`)**
  - Performs Partial Least Squares Discriminant Analysis (PLS-DA) to statistically analyze and visualize shape variations between healthy controls and epilepsy patients.

### 3. Machine Learning Modeling (`Model/`)
- Contains the scripts and results for training predictive models based on the SPHARM morphological features.
- Supports cross-dataset evaluation (e.g., Ds004469 and Ds005602) and various data augmentation techniques.
- Generates detailed statistical reports, P-value summaries, and model evaluation metrics.

### 4. Desktop Application (`DesktopApp/`)
- A Graphical User Interface (GUI) built with Python (PyQt/PySide).
- Allows medical researchers and users to easily select input folders, configure parameters, run the pipelines without touching the command line, and visualize the 3D results seamlessly.

## 🚀 Quick Start

### Preprocessing
To extract the hippocampus from raw MRI scans:
```bash
python run_pipeline.py --input_dir /path/to/raw/mri/
```

### Shape Analysis
To run the full ICP and SPHARM pipeline on your extracted dataset:
```bash
# Double-click the file on Windows or run via terminal:
run_everything.bat
```
*(This will open a folder picker prompt to select your dataset, and automatically process all steps via SlicerSALT).*

## 🛠️ Dependencies
- **Python 3.x** (`numpy`, `scipy`, `pandas`, `vtk`, `matplotlib`)
- **FastSurfer** (for automatic deep-learning brain segmentation)
- **SlicerSALT 6.0+** (Required for the ICP and SPHARM-PDM shape analysis tools)