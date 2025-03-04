import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import esm
import os
def prepare_data(file_name, model, batch_converter, device, save_path=None):
    """
    Prepares data by loading sequences, generating ESM embeddings, and optionally saving them.

    Args:
        file_name (str): Path to the CSV file containing sequences and labels.
        model: Pre-trained ESM model for generating embeddings.
        batch_converter: ESM batch converter for processing sequences.
        device: Device (CPU or GPU) to use for computation.
        save_path (str, optional): Path to save the embeddings and labels. If None, embeddings are not saved.

    Returns:
        pooled_embeddings (np.array): Pooled ESM embeddings (e.g., [CLS] token embeddings).
        token_embeddings (np.array): Token-level ESM embeddings.
        labels (list): Corresponding labels for the sequences.
    """
    print(f'Preparing data from {file_name}')
    
    # Check if file exists
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Data file not found: {file_name}")
    
    # Load the CSV file
    df = pd.read_csv(file_name)
    
    # Validate required columns
    required_columns = ["sequence", "label", "id"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {file_name}")

    # Extract sequences and labels
    sequences = df["sequence"].tolist()
    labels = df["label"].tolist()
    ids = df["id"].tolist()

    # Prepare data for ESM
    data = list(zip(ids, sequences))

    # Generate ESM embeddings in smaller batches
    batch_size = 16  # Adjust based on GPU memory
    pooled_embeddings = []
    token_embeddings = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            # Pooled embeddings (e.g., [CLS] token)
            pooled_batch_embeddings = results["representations"][33][:, 0, :].cpu().numpy()
            pooled_embeddings.append(pooled_batch_embeddings)

            # Token-level embeddings
            token_batch_embeddings = results["representations"][33].cpu().numpy()
            token_embeddings.append(token_batch_embeddings)

    # Concatenate pooled embeddings
    pooled_embeddings = np.concatenate(pooled_embeddings, axis=0)
    
    # Convert token embeddings to tensors and pad to same length
    token_embeddings_flat = []
    for batch_embs in token_embeddings:
        for seq_emb in batch_embs:
            token_embeddings_flat.append(torch.tensor(seq_emb))
            
    # Then pad the sequences
    padded_token_embeddings = pad_sequence(token_embeddings_flat, batch_first=True, padding_value=0)
    token_embeddings = padded_token_embeddings.cpu().numpy()

    # Save embeddings and labels if save_path is provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save files
        np.save(f"{save_path}_pooled_embeddings.npy", pooled_embeddings)
        np.save(f"{save_path}_token_embeddings.npy", token_embeddings)
        np.save(f"{save_path}_labels.npy", np.array(labels))
        print(f"Embeddings and labels saved to {save_path}_pooled_embeddings.npy, {save_path}_token_embeddings.npy, and {save_path}_labels.npy")

    return pooled_embeddings, token_embeddings, labels

def load_embeddings(embeddings_path, token_embeddings_path, labels_path):
    """
    Loads embeddings and labels from disk.

    Args:
        embeddings_path (str): Path to the saved embedding of the CLS token 
        (i.e. pooled embedding of entire sequence).
        token_embeddings_path (str): Path to the saved embeddings of all the tokens in the sequence.
        labels_path (str): Path to the saved labels file.

    Returns:
        embeddings (np.array): Loaded CLS embedding.
        token_embeddings (np.array): Loaded token embeddings.
        labels (np.array): Loaded labels.
    """
    print(f'Loading embeddings from {embeddings_path}')
    
    # Check if files exist
    for path in [embeddings_path, token_embeddings_path, labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # Load files
    embeddings = np.load(embeddings_path)
    token_embeddings = np.load(token_embeddings_path)
    labels = np.load(labels_path)
    
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    print(f"Loaded token embeddings with shape: {token_embeddings.shape}")
    print(f"Loaded {len(labels)} labels")
    
    return embeddings, token_embeddings, labels

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plots a confusion matrix.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        class_names (list): Names of the classes.
        save_path (str, optional): Path to save the plot. If None, plot is not saved.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
