from utils import plot_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import shap
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
from torch.optim.lr_scheduler import ReduceLROnPlateau
import traceback
class DNNClassifier(nn.Module):
    """
    A Deep Neural Network classifier for protein sequence variant classification.
    """
    def __init__(self, input_dim, hidden_dim):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output layer
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation, since BCEWithLogitsLoss expects raw logits
        return x
def train_dnn(embeddings_train, labels_train, embeddings_eval, labels_eval, device, 
              batch_size=32, hidden_dim=128, learning_rate=0.001, weight_decay=0.01, 
              num_epochs=100, patience=5):
    """
    Trains the DNN classifier with early stopping and learning rate scheduling.

    Args:
        embeddings_train (np.array): Training embeddings.
        labels_train (list): Training labels.
        embeddings_eval (np.array): Validation embeddings.
        labels_eval (list): Validation labels.
        device: Device (CPU or GPU) to use for training.
        batch_size (int): Batch size for training.
        hidden_dim (int): Size of hidden layers.
        learning_rate (float): Initial learning rate.
        weight_decay (float): L2 regularization strength.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.

    Returns:
        model: The trained DNN model.
    """
    # Convert data to PyTorch tensors
    print("Training DNN classifier")
    X_train = torch.tensor(embeddings_train, dtype=torch.float32)
    y_train = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)  # Add extra dimension for BCEWithLogitsLoss
    X_val = torch.tensor(embeddings_eval, dtype=torch.float32)
    y_val = torch.tensor(labels_eval, dtype=torch.float32).unsqueeze(1)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = X_train.shape[1]
    model = DNNClassifier(input_dim, hidden_dim).to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Mixed Precision Training
    scaler = GradScaler()

    # Early stopping setup
    best_val_loss = float("inf")
    counter = 0
    early_stop = False

    # Training metrics history
    train_losses = []
    val_losses = []
    accuracies = []
    f1_scores = []
    
    # Training loop
    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping triggered after {epoch} epochs")
            break
            
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass with mixed precision
            with autocast():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Use mixed precision for validation too
                with autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()

                # Convert logits to probabilities
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        accuracies.append(accuracy)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        f1_scores.append(f1)
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
            auc_metric = f" | AUC: {auc:.4f}"
        except ValueError:
            # Handle case where there might be only one class in the batch
            auc_metric = ""

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Acc: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}{auc_metric}")

        # Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'models/best_dnn_model.pth')
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                early_stop = True

    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    
    plt.tight_layout()
    plt.savefig('figures/dnn_training_curves.png')
    plt.show()
    
    # Load the best model for returning
    model.load_state_dict(torch.load('models/best_dnn_model.pth'))
    return model

def test_dnn(model, embeddings_test, labels_test, device, batch_size=32):
    """
    Evaluates the DNN model on the test set and explains predictions using SHAP.

    Args:
        model: Trained DNN model.
        embeddings_test (np.array): Test pooled embeddings.
        labels_test (list): Test labels.
        device: Device (CPU or GPU) to use for evaluation.
        batch_size (int): Batch size for evaluation.
    """
    # Convert test data to PyTorch tensors
    X_test = torch.tensor(embeddings_test, dtype=torch.float32)
    y_test = torch.tensor(labels_test, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            with autocast():
                outputs = model(batch_X)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Compute test metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Test AUC: {auc:.4f}")
    except ValueError:
        print("Could not calculate AUC (possibly only one class in test set)")
    
    print(f"DNN Test Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(
            all_labels,
            all_preds,
            class_names=["Benign", "Pathogenic"],
            model_name='DNN',
            save_path="figures/DNN_confusion_matrix.png"
        )
