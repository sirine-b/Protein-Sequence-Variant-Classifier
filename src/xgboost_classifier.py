import xgboost as xgb
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_confusion_matrix
import shap

def train_xgboost(embeddings_train, labels_train, embeddings_eval, labels_eval):
    """
    Train an XGBoost classifier on ESM embeddings.

    Args:
        embeddings_train (np.array): Training embeddings
        labels_train (list): Training labels
        embeddings_eval (np.array): Validation embeddings
        labels_eval (list): Validation labels

    Returns:
        model: Trained XGBoost model
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV

        print("Training XGBoost classifier")

        # Create DMatrix objects
        dtrain = xgb.DMatrix(embeddings_train, label=labels_train)
        deval = xgb.DMatrix(embeddings_eval, label=labels_eval)

        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Initial XGBoost model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            verbose=1
        )

        # Fit the model
        grid_search.fit(
            embeddings_train,
            labels_train,
            eval_set=[(embeddings_eval, labels_eval)],
            verbose=False
        )

        # Get best model and parameters
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

        # Save the model
        best_model.save_model("models/best_xgboost_model.json")
        print("Best XGBoost model saved to 'models/best_xgboost_model.json'")

        return best_model

    except ImportError:
        print("XGBoost is not installed. Please install it with: pip install xgboost")
        return None

def test_xgboost(model, embeddings_test, labels_test):
    """
    Evaluate XGBoost model on test data.

    Args:
        model: Trained XGBoost model
        embeddings_test (np.array): Test embeddings
        labels_test (list): Test labels
    """
    try:
        if model is None:
            print("No XGBoost model provided for testing")
            return

        print("Evaluating XGBoost model on test data")

        # Make predictions
        y_pred = model.predict(embeddings_test)
        y_prob = model.predict_proba(embeddings_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(labels_test, y_pred)
        precision = precision_score(labels_test, y_pred)
        recall = recall_score(labels_test, y_pred)
        f1 = f1_score(labels_test, y_pred)

        print(f"XGBoost Test Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Plot confusion matrix
        plot_confusion_matrix(
            labels_test,
            y_pred,
            class_names=["Benign", "Pathogenic"],
            model_name='XGBoost',
            save_path="figures/xgboost_confusion_matrix.png"
        )

        # SHAP Analysis for feature importance
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(embeddings_test[:100])
            plt.figure(figsize=(10, 6))  
            shap.summary_plot(
                shap_values, 
                embeddings_test[:100], 
                feature_names=[f"ESM_Embedding_Dim_{i}" for i in range(embeddings_test.shape[1])], 
                show=False,  # Prevents automatic display
                title="SHAP Values for XGBoost Model"
            )

            # Save the plot before showing
            plt.savefig("figures/xgboost_shap_values.png", dpi=300, bbox_inches='tight')  
            plt.show()  # Show if running interactively
            plt.close()  # Close the figure to prevent duplication issues
        except Exception as e:
            print(f"SHAP analysis for XGBoost failed: {e}")

    except ImportError:
        print("XGBoost is not installed. Please install it with: pip install xgboost")
