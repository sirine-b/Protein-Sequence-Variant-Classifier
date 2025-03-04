import torch
import esm
from utils import prepare_data, load_embeddings
from xgboost_classifier import train_xgboost, test_xgboost
import xgboost as xgb
from dnn_classifier import train_dnn, test_dnn, DNNClassifier
import os
import pandas as pd

def main():
    """
    Main function to orchestrate the workflow.
    """
    # Step 1: Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 2: Load a smaller ESM model
    try:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model = model.to(device)
        model.eval()
        print("ESM model loaded successfully")
    except Exception as e:
        print(f"Error loading ESM model: {e}")
        return

    # Create consistent path names
    train_base_path = "embeddings/train"
    val_base_path = "embeddings/val"
    test_base_path = "embeddings/test"

    # Step 3: Prepare and load data
    train_embeddings_path = f"{train_base_path}_pooled_embeddings.npy"
    train_token_embeddings_path = f"{train_base_path}_token_embeddings.npy"
    train_labels_path = f"{train_base_path}_labels.npy"

    try:
        if all(os.path.exists(p) for p in [train_embeddings_path, train_token_embeddings_path, train_labels_path]):
            # Load precomputed embeddings
            embeddings_train, token_embeddings_train, labels_train = load_embeddings(
                train_embeddings_path, train_token_embeddings_path, train_labels_path
            )
        else:
            # Generate and save embeddings
            embeddings_train, token_embeddings_train, labels_train = prepare_data(
                "data/df_tp53_train.csv", model, batch_converter, device, save_path=train_base_path
            )
    except Exception as e:
        print(f"Error loading or generating training embeddings: {e}")
        return

    # Repeat for validation data
    val_embeddings_path = f"{val_base_path}_pooled_embeddings.npy"
    val_token_embeddings_path = f"{val_base_path}_token_embeddings.npy"
    val_labels_path = f"{val_base_path}_labels.npy"

    try:
        if all(os.path.exists(p) for p in [val_embeddings_path, val_token_embeddings_path, val_labels_path]):
            embeddings_eval, token_embeddings_eval, labels_eval = load_embeddings(
                val_embeddings_path, val_token_embeddings_path, val_labels_path
            )
        else:
            embeddings_eval, token_embeddings_eval, labels_eval = prepare_data(
                "data/df_tp53_eval.csv", model, batch_converter, device, save_path=val_base_path
            )
    except Exception as e:
        print(f"Error loading or generating validation embeddings: {e}")
        return

    # Repeat for test data
    test_embeddings_path = f"{test_base_path}_pooled_embeddings.npy"
    test_token_embeddings_path = f"{test_base_path}_token_embeddings.npy"
    test_labels_path = f"{test_base_path}_labels.npy"

    try:
        if all(os.path.exists(p) for p in [test_embeddings_path, test_token_embeddings_path, test_labels_path]):
            embeddings_test, token_embeddings_test, labels_test = load_embeddings(
                test_embeddings_path, test_token_embeddings_path, test_labels_path
            )
        else:
            embeddings_test, token_embeddings_test, labels_test = prepare_data(
                "data/df_tp53_test.csv", model, batch_converter, device, save_path=test_base_path
            )
    except Exception as e:
        print(f"Error loading or generating test embeddings: {e}")
        return

    # Step 4: Check for pre-trained models
    dnn_model_path = "models/best_dnn_model.pth"
    xgboost_model_path = "models/best_xgboost_model.json"

    if os.path.exists(dnn_model_path) and os.path.exists(xgboost_model_path):
        # Load pre-trained models and test them
        print("Pre-trained models found. Loading and testing...")

        # Load DNN model
        dnn_classifier = DNNClassifier(input_dim=embeddings_train.shape[1], hidden_dim=128).to(device)
        dnn_classifier.load_state_dict(torch.load(dnn_model_path))
        dnn_classifier.eval()
        print("Loaded pre-trained DNN model.")

        # Test DNN model
        test_dnn(dnn_classifier, embeddings_test, token_embeddings_test, labels_test, device)

        # Load XGBoost model
        xgboost_classifier = xgb.XGBClassifier()
        xgboost_classifier.load_model(xgboost_model_path)
        print("Loaded pre-trained XGBoost model.")

        # Test XGBoost model
        test_xgboost(xgboost_classifier, embeddings_test, labels_test)

    else:
        # Train and evaluate DNN
        print("Pre-trained models not found. Training models...")
        try:
            dnn_classifier = train_dnn(embeddings_train, labels_train, embeddings_eval, labels_eval, device)
            test_dnn(dnn_classifier, embeddings_test, token_embeddings_test, labels_test, device)
        except Exception as e:
            print(e)

        # Train and evaluate XGBoost
        try:
            xgboost_classifier = train_xgboost(embeddings_train, labels_train, embeddings_eval, labels_eval)
            test_xgboost(xgboost_classifier, embeddings_test, labels_test)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()