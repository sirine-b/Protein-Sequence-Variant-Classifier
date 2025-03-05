# Protein Variant Pathogenicity Prediction

This project focuses on predicting whether protein variants are **benign** or **pathogenic** using deep neural network (**DNN**) and extreme gradient boosting (**XGBoost**) models. The models are trained on embeddings generated from protein sequences using the **ESM** (Evolutionary Scale Modeling) model, a state-of-the-art **Transformer protein language model**.

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
- **ESM Embeddings**: Utilises the ESM model to generate both pooled (CLS token) and token-level embeddings for protein sequences.
- **Deep Neural Network (DNN)**: A custom DNN model with batch normalization, dropout, and leaky ReLU activation for binary classification.
- **XGBoost**: A gradient boosting model optimised using grid search for hyperparameter tuning.
- **SHAP Explainability**: Provides feature importance and model interpretability using SHAP (SHapley Additive exPlanations).
- **Training Curves and Confusion Matrices**: Visualisations of training progress and model performance.
- **Uses data from [TP53 Protein Variants](https://huggingface.co/datasets/sequential-lab/TP53_protein_variants/tree/main)**: The training, validation and testing data was  downloaded from the TP53_protein_variants dataset on HuggingFace.
- **Embedding Interpretability Analysis**: Correlates learned ESM embedding dimensions with features of the protein sequences to provide insights beyond traditional SHAP analysis (which only tells us which dimensions are most important to the prediction but not what these dimensions relate to concretely).
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
  ├── utils.py               # Helper functions for data processing and visualisation
  ├── dnn_classifier.py      # Deep Neural Network model training & testing
  └── xgboost_classifier.py  # XGBoost model training & testing
├── pretrained/              # Pretrained models and already generated embeddings to skip embedding and training stages
  ├── models_to_download.py        # Pretrained models 
  ├── embeddings_to_download.py    # Already generated embeddings for provided data (TP53 CSV files) 
  
 # Below folders are created once the model runs #
├── embeddings/             # Saved computed embeddings to save time and prevent having to compute them at each run
└── models/                 # Saved models and results
 ```
## Installation
### 1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/protein-variant-prediction.git
   cd protein-variant-prediction
   ```
### 2. Install the required libraries
   ```bash
   pip install -r requirements.txt
   ```
## Usage

### 1. Prepare Your Data

#### Case 1: Using Provided Data
- The repository already includes example data for training, validation, and testing in the `data/` folder:
  - `data/df_tp53_train.csv`: Training data.
  - `data/df_tp53_eval.csv`: Validation data.
  - `data/df_tp53_test.csv`: Test data.
- Each CSV file contains the following columns:
  - `id`: Unique identifier for each variant.
  - `sequence`: Protein sequence.
  - `label`: Binary classification (`0` for benign, `1` for pathogenic).

#### Case 2: Using Your Own Data
If you want to use your own data, follow these steps:
1. Format your data as a CSV file with the following columns:
   - `id`: Unique identifier for each variant.
   - `sequence`: Protein sequence.
   - `label`: Binary classification (`0` for benign, `1` for pathogenic).
2. Add your CSV files to the `data/` folder.
3. Update the file names in the code:
   - For training data, update the file name in **line 47** of `main.py`.
   - For validation data, update the file name in **line 65** of `main.py`.
   - For test data, update the file name in **line 83** of `main.py`.

### 2. Running the Script

#### Case 1: Train and Test your own Models

To train and evaluate the models, run the following command:

```bash
python main.py
```
This will: 
1. Load the ESM-2 model
2. Generate embeddings for your protein sequences.
3. Train the DNN classifier and XGBoost model on the obtained embeddings.
4. Evaluate performance of both classifiers on the test set.
5. Save results, plots, and models.

#### Case 2: Test the Pretrained Models with pre-Generated Embeddings
If you want to skip training and embedding and directly start using the pre-trained models provided in the repository:

##### 1. Download the pre-trained models and embeddings (generated for the TP53 data files provided under data/ folder) from the pretrained/ folder.

##### 2. Place the downloaded files in the appropriate folders:

- Trained models: models/

- Embeddings: embeddings/

##### 3. Run the main.py script:

```bash
python main.py
```

**_Note_**: The main.py script checks whether the pretrained models and embeddings are present in the correct folders and under the correct name. If yes (i.e. when you've done steps 1 and 2), then it will load the pretrained models and test them on the downloaded embeddings. If not (i.e. case 1) , it will generate embeddings based on given data and train models from scratch.

### 3. Check the results
   
   After running the pipeline, check:

* models/best_dnn_model.pth → Best trained DNN model.

* models/best_xgboost_model.json → Best trained XGBoost model.

* figures/training_curves.png → Training loss and accuracy.

* figures/xgboost_confusion_matrix.png → Confusion matrix of XGBoost classifier on test data.
  
* figures/dnn_confusion_matrix.png → Confusion matrix of DNN classifier on test data.
  
* figures/xgboost_feature_importance.png → SHAP Analysis for XGBoost classifier (i.e. SHAP scores of most important embedding dimensions).
  
* figures/all_features_emb_correlation_analysis.png → Visualisation of correlation between the protein sequence features and embedding dimensions

* figures/top_features_emb_correlation_analysis.png → Visualisation of correlation between the top 10 most important protein sequence features and first 50 embedding dimensions

## Results
### Metrics
Below are the results obtained whether the trained models were tested on the test dataset provided in the data/ folder: 

| Model   | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| DNN     | 0.8028   | 1.0000    | 0.6500 | 0.7879   |
| XGBoost | 0.9155   | 0.8864    | 0.9750 | 0.9286   |



<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/9907f4fc-e742-4c08-abe5-2082b63156c3" alt="Image 1" style="width: 48%; margin-right: 2%;">
  <img src="https://github.com/user-attachments/assets/cb045978-b069-49e5-91d9-e68185c98750" alt="Image 2" style="width: 48%;">
</div>
### SHAP Analysis and Interpretability
The reason I performed SHAP analysis was to identify which dimensions of the ESM embeddings are most influential in predicting protein variant pathogenicity and understand better the outputs/predictions made by the classifiers. However, these dimensions are not inherently interpretable because they arise from a self-supervised learning process. In othe words, unlike predefined biological features (e.g. length of sequence, GC content etc), they capture abstract, non-linear representations of protein sequences. As such, simply learning that embedding dimensions 12 and 657 for example contributed the most to the prediction doesn't necessarily provide meaningful biological insights. To address this, I attempted to analyse correlations between embedding dimensions and biologically relevant protein sequence features (i.e., amino acid composition, hydrophobicity, GC content). This helped relate machine-learnt features to tangible biochemical properties, and therefore offered insights into what the model may be prioritising in terms of protein sequence features this time around (much more useful for interpretability). However, it is important to keep in mind tht I only explored a few biochemical features and these may not be the most relevant or useful ones to correlate to the ESM embedding dimensions. As such, although I was able to learn about some correlations between embedding dimensions and protein sequence features, this is an aspect that still needs to be worked on to further improve interpretability.

![image](https://github.com/user-attachments/assets/47a809ff-d704-477c-b6e8-f14083059327)
![image](https://github.com/user-attachments/assets/1d25a590-b3c7-4663-bd69-a59135eb0366)
![image](https://github.com/user-attachments/assets/d3497f0a-bc1d-4a9d-8775-63c7bbcb0331)


