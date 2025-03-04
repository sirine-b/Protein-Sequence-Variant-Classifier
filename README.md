# Protein Variant Pathogenicity Prediction

This project focuses on predicting whether protein variants are **benign** or **pathogenic** using deep learning (DNN) and gradient boosting (XGBoost) models. The models are trained on embeddings generated from protein sequences using the **ESM (Evolutionary Scale Modeling)** model, a state-of-the-art protein language model.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction
Protein variants can have significant implications for human health, with some variants being benign and others pathogenic. This project leverages machine learning to classify protein variants based on their sequence data. The embeddings are generated using the **ESM model**, and the classification is performed using both **Deep Neural Networks (DNN)** and **XGBoost**.

## Features
- **ESM Embeddings**: Utilizes the ESM model to generate both pooled (CLS token) and token-level embeddings for protein sequences.
- **Deep Neural Network (DNN)**: A custom DNN model with batch normalization, dropout, and leaky ReLU activation for binary classification.
- **XGBoost**: A gradient boosting model optimised using grid search for hyperparameter tuning.
- **SHAP Explainability**: Provides feature importance and model interpretability using SHAP (SHapley Additive exPlanations).
- **Training Curves and Confusion Matrices**: Visualisations of training progress and model performance.
- **Uses data from [TP53 Protein Variants](https://huggingface.co/datasets/sequential-lab/TP53_protein_variants/tree/main)**: The training, validation and testing data was  downloaded from the TP53_protein_variants dataset on HuggingFace.

## Model Details
**Deep Neural Network**

* 3-layer architecture with batch normalization
* LeakyReLU activation and dropout for regularization
* Mixed precision training for better performance
* Learning rate scheduling
* Performance metrics: accuracy, precision, recall, F1 score, AUC

**XGBoost**

* Hyperparameter optimization using GridSearchCV
* Binary logistic objective function
* Feature importance visualisation
* SHAP analysis for model interpretability
  
## Project Structure
 ``` bash
├── data/                   # CSV files containing sequences and labels
├── src/
  ├── main.py                # Orchestrates the full pipeline
  ├── utils.py               # Helper functions for data processing and visualization
  ├── dnn_classifier.py      # Deep Neural Network model training & testing
  └── xgboost_classifier.py  # XGBoost model training & testing
  
 # Below folders are created once the model runs #
├── embeddings/             # Saved computed embeddings to save time and prevent having to compute them at each run
└──  models/                # Saved models and results
 ```
## Installation
**1. Clone the repository:**
   ```bash
   git clone https://github.com/your-username/protein-variant-prediction.git
   cd protein-variant-prediction
   ```
**2. Install the required libraries**
   ```bash
   pip install -r requirements.txt
   ```
## Usage 
**1. Prepare your data**

   _**Note**_: Data for training, validation and testing is already included in the repository (under the data/ folder) in the form of CSV files.

   However, if you wish you add your own data, feel free to do so. Make sure to:
- Format it as a CSV file with_sequence with the following columns:
    - id: Unique identifier for each variant
    - sequence: Protein sequence
    - label: Binary classification (0 for benign, 1 for pathogenic),
- Add CSV files to the data/ folder
- Update the code with the corresponding file names given (i.e. specifically need to update lines 47, 65, and 83 for the training, validation and testing datasets respectively).
   
**2. Run the full pipeline**
   ```bash
   python main.py
   ```
   
   This will:
    1. Load the ESM-2 model
    2. Generate embeddings for your protein sequences.
    3. Train the DNN classifier and XGBoost model on the obtained embeddings.
    4. Evaluate performance of both classifiers on the test set.
    5. Save results, plots, and models.
       
**3. Check the results**
   
   After running the pipeline, check:

* models/best_dnn_model.pth → Best trained DNN model.

* models/best_xgboost_model.json → Best trained XGBoost model.

* models/training_curves.png → Training loss and accuracy.

* models/confusion_matrix.png → Test set confusion matrix.
  
* models/xgboost_feature_importance.png → Feature importance plot for XGBoost.

## Results
| Model   | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| DNN     | XX.XX%   | XX.XX%    | XX.XX% | XX.XX%   |
| XGBoost | XX.XX%   | XX.XX%    | XX.XX% | XX.XX%   |
