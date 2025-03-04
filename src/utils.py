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

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path=None):
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
    plt.title(f"{model_name} Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def detailed_correlation_analysis(embeddings, sequences):
    """
    Provide more detailed view of correlations, including positive and negative relationships
    
    Args:
        embeddings (np.array): ESM embeddings
        sequences (list): Protein sequences
    
    Returns:
        DataFrame: Correlation matrix between embedding dimensions and sequence features
    """
    # Ensure numpy and pandas are imported
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Calculate sequence features
    sequence_features = calculate_sequence_features(sequences)
    
    # Convert embeddings to DataFrame
    # Ensure embeddings is a 2D numpy array
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    
    # Create DataFrames
    feature_df = pd.DataFrame(sequence_features)
    embedding_df = pd.DataFrame(embeddings)
    
    # Compute correlation matrix
    # Transpose embedding_df to have embedding dimensions as columns
    full_correlation_matrix = feature_df.corrwith(embedding_df.T)
    
    # Convert to DataFrame for easier manipulation
    full_correlation_matrix = pd.DataFrame(full_correlation_matrix)
    
    # Plot heatmap of correlations
    plt.figure(figsize=(12, 8))
    
    # Check if correlation matrix is not empty
    if not full_correlation_matrix.empty and len(full_correlation_matrix) > 0:
        sns.heatmap(full_correlation_matrix, cmap='coolwarm', center=0, 
                    annot=True, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation between Sequence Features and Embedding Dimensions')
        plt.xlabel('Embedding Dimensions')
        plt.ylabel('Sequence Features')
        plt.tight_layout()
        plt.savefig('full_embedding_correlation_heatmap.png')
        plt.show()
    else:
        print("Correlation matrix is empty. Check input data.")
    
    return full_correlation_matrix
    
# def calculate_sequence_features(sequences):
#     """
#     Calculate various features for protein sequences
    
#     Args:
#         sequences (list): List of protein sequences
    
#     Returns:
#         dict: Dictionary of sequence-level features
#     """
#     # Amino acid composition
#     aa_freq = {}
#     for seq in sequences:
#         for aa in 'ACDEFGHIKLMNPQRSTVWY':
#             if aa not in aa_freq:
#                 aa_freq[aa] = []
#             aa_freq[aa].append(seq.count(aa) / len(seq))
    
#     # Aggregate features
#     sequence_features = {
#         'length': [len(seq) for seq in sequences],
#         'gc_content': [seq.count('G') + seq.count('C') / len(seq) for seq in sequences],
#         'hydrophobicity': [calculate_hydrophobicity(seq) for seq in sequences],
#         **{f'{aa}_freq': np.array(freq) for aa, freq in aa_freq.items()}
#     }
    
#     return sequence_features

# def calculate_hydrophobicity(sequence):
#     """Basic hydrophobicity calculation using Kyte-Doolittle scale"""
#     hydrophobicity_scale = {
#         'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
#         'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
#         'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
#         'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': 2.3
#     }
#     return sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / len(sequence)

# def analyze_embedding_correlations(embeddings, sequences, top_n=10):
#     """
#     Analyze correlations between ESM embedding dimensions and sequence characteristics
    
#     Args:
#         embeddings (np.array): ESM embeddings
#         sequences (list): Protein sequences
#         top_n (int): Number of top correlations to display
#     """
#     # Calculate sequence features
#     sequence_features = calculate_sequence_features(sequences)
    
#     # Convert to DataFrame
#     feature_df = pd.DataFrame(sequence_features)
#     embedding_df = pd.DataFrame(embeddings)
    
#     # Compute correlations
#     correlation_results = {}
#     for feature in feature_df.columns:
#         # Compute correlation between embedding dimensions and the feature
#         correlations = embedding_df.corrwith(feature_df[feature])
        
#         # Store absolute top correlations
#         correlation_results[feature] = correlations.abs().nlargest(top_n)
    
#     # Visualization
#     plt.figure(figsize=(15, 5 * len(correlation_results)))
    
#     for i, (feature, top_corr) in enumerate(correlation_results.items(), 1):
#         plt.subplot(len(correlation_results), 1, i)
#         sns.barplot(x=top_corr.index, y=top_corr.values, palette='viridis')
#         plt.title(f'Top {top_n} Embedding Dims Correlated with {feature}')
#         plt.xticks(rotation=90)
#         plt.xlabel('Embedding Dimensions')
#         plt.ylabel('Absolute Correlation')
    
#     plt.tight_layout()
#     plt.savefig('embeddings/embedding_feature_correlations.png')
#     plt.show()
    
#     # Return detailed correlation results
#     return correlation_results

# # Optional: Additional analysis method to show signed correlations
# def detailed_correlation_analysis(embeddings, sequences):
#     """
#     Provide more detailed view of correlations, including positive and negative relationships
    
#     Args:
#         embeddings (np.array): ESM embeddings
#         sequences (list): Protein sequences
    
#     Returns:
#         DataFrame: Correlation matrix between embedding dimensions and sequence features
#     """
#     # Ensure numpy and pandas are imported
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     # Calculate sequence features
#     sequence_features = calculate_sequence_features(sequences)
    
#     # Convert embeddings to DataFrame
#     # Ensure embeddings is a 2D numpy array
#     if not isinstance(embeddings, np.ndarray):
#         embeddings = np.array(embeddings)
    
#     # Create DataFrames
#     feature_df = pd.DataFrame(sequence_features)
#     embedding_df = pd.DataFrame(embeddings)
    
#     # Compute correlation matrix
#     # Transpose embedding_df to have embedding dimensions as columns
#     full_correlation_matrix = feature_df.corrwith(embedding_df.T)
    
#     # Convert to DataFrame for easier manipulation
#     full_correlation_matrix = pd.DataFrame(full_correlation_matrix)
    
#     # Plot heatmap of correlations
#     plt.figure(figsize=(12, 8))
    
#     # Check if correlation matrix is not empty
#     if not full_correlation_matrix.empty and len(full_correlation_matrix) > 0:
#         sns.heatmap(full_correlation_matrix, cmap='coolwarm', center=0, 
#                     annot=True, cbar_kws={'label': 'Correlation'})
#         plt.title('Correlation between Sequence Features and Embedding Dimensions')
#         plt.xlabel('Embedding Dimensions')
#         plt.ylabel('Sequence Features')
#         plt.tight_layout()
#         plt.savefig('full_embedding_correlation_heatmap.png')
#         plt.show()
#     else:
#         print("Correlation matrix is empty. Check input data.")
    
#     return full_correlation_matrix
def calculate_sequence_features(sequences):
    """
    Calculate various features from protein sequences.
    
    Args:
        sequences (list): List of protein sequences
        
    Returns:
        dict: Dictionary of sequence features
    """
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    features = {
        'length': [],
        'hydrophobic_ratio': [],
        'charged_ratio': [],
    }
    
    # Add individual amino acid frequencies
    for aa in amino_acids:
        features[f'{aa}_freq'] = []
    
    hydrophobic = 'AILMFWV'
    charged = 'DEKR'
    
    for seq in sequences:
        # Sequence length
        length = len(seq)
        features['length'].append(length)
        
        # Amino acid frequencies
        for aa in amino_acids:
            freq = seq.count(aa) / length if length > 0 else 0
            features[f'{aa}_freq'].append(freq)
        
        # Hydrophobic ratio
        hydrophobic_count = sum(seq.count(aa) for aa in hydrophobic)
        features['hydrophobic_ratio'].append(hydrophobic_count / length if length > 0 else 0)
        
        # Charged ratio
        charged_count = sum(seq.count(aa) for aa in charged)
        features['charged_ratio'].append(charged_count / length if length > 0 else 0)
    
    return features
def detailed_correlation_analysis(embeddings, sequences):
    """
    Conducts a correlation analysis between sequence features and embeddings.
    For visualisation: 
    - A figure that shows only the top 10 most correlated features and only the first 50 embedding dimensions
    - A figure that shows all features and all embedding dimensions
    Args:
        embeddings (np.array): ESM embeddings
        sequences (list): Protein sequences
    
    Returns:
        DataFrame: Correlation matrix between embedding dimensions and sequence features
    """
    # Calculate sequence features
    sequence_features = calculate_sequence_features(sequences)
    
    # Convert embeddings to DataFrame
    embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
    
    # Convert sequence features to DataFrame
    feature_df = pd.DataFrame(sequence_features)
    
    # Calculate mean absolute correlation for each feature across all embedding dimensions
    mean_correlations = {}
    for feature in feature_df.columns:
        correlations = np.abs([feature_df[feature].corr(embedding_df[emb]) for emb in embedding_df.columns])
        mean_correlations[feature] = np.mean(correlations)
    
    # Sort features by mean absolute correlation
    sorted_features = sorted(mean_correlations.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:10]]  # Take top 10 features
    
    # Calculate correlations for top features
    correlations = pd.DataFrame()
    for feature in top_features:
        for emb in embedding_df.columns[:50]:  # Take first 50 embedding dimensions
            corr = feature_df[feature].corr(embedding_df[emb])
            correlations.loc[feature, emb] = corr
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot heatmap with improved readability
    sns.heatmap(correlations, 
                cmap='coolwarm', 
                center=0,
                vmin=-0.5, 
                vmax=0.5,
                xticklabels=5,  # Show every 5th embedding dimension
                yticklabels=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    # Improve labels and title
    plt.title('Correlation between top 10 Sequence Features and First 50 Embedding Dimensions', 
              pad=20, 
              fontsize=12)
    plt.xlabel('Embedding Dimensions (first 50)', fontsize=10)
    plt.ylabel('Sequence Features', fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig('embeddings/top_features_emb_correlation_analysis.png', dpi=300, bbox_inches='tight')
    
    
    # Second figure that shows all sequence features and all embedding dimensions
    # Calculate correlations
    correlations = pd.DataFrame()
    for feature in feature_df.columns:
        for emb in embedding_df.columns:
            corr = feature_df[feature].corr(embedding_df[emb])
            correlations.loc[feature, emb] = corr
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(correlations, 
                cmap='coolwarm', 
                center=0,
                xticklabels=False,  # Hide embedding dimension labels for clarity
                yticklabels=True,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Correlation between All Sequence Features and Embedding Dimensions')
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Sequence Features')
    plt.tight_layout()
    plt.savefig('embeddings/all_features_emb_correlation_analysis.png', dpi=300, bbox_inches='tight')
    return correlations
