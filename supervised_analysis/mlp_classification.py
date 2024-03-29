import click
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import xgboost as xgb
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm 
from itertools import product
from autogluon.tabular import TabularPredictor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
import optuna
import os

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dr):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dr)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out) 
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
class MLP_complex(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dr):
        super(MLP_complex, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(dr),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(dr),
            nn.Linear(hidden_sizes[1], num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

@click.command()
@click.option('input_dataset','-i', type=click.Path(exists=True))
@click.option('--model_type', '-m', type=click.Choice(['mlp', 'mlp_complex', 'xgboost', "autogluon", "svm", "mlp_hyperparameter"], case_sensitive=False), required=True)
@click.option('--hidden_size', '-h', default = "100", help='Number of hidden layers.')
@click.option('--epochs','-e', default=20, type=int, help='Number of epochs for training.')
@click.option('--lr', '-l', default=0.01, type=float, help='Learning rate.')
@click.option('--wd', '-w', default=0.00005, type=float, help='Weight decay.')
@click.option('--dr', '-d', default=0.5, type=float, help='Dropout rate.')
@click.option('--output', '-o', type=str, help='Folder for results.')

def train_model(input_dataset, model_type, epochs, lr, hidden_size, dr, wd, output):
    if "BF" in output:
        label_codes = {8: 'retinoid receptor agonist',
                        9: 'topoisomerase inhibitor',
                        0: 'ATPase inhibitor',
                        10: 'tubulin polymerization inhibitor',
                        6: 'dmso',
                        7: 'protein synthesis inhibitor',
                        5: 'PARP inhibitor',
                        1: 'Aurora kinase inhibitor',
                        3: 'HSP inhibitor',
                        2: 'HDAC inhibitor',
                        4: 'JAK inhibitor'}
    else:
        label_codes = {0: "AKT", 1: "CDK", 2: "DMSO", 3: "HDAC", 4: "MAPK", 5: "PARP", 6: "TUB"}  
    # Load the dataset
    df = pd.read_csv(input_dataset)
    X = df.drop(['label'], axis=1).values
    y = df['label'].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42) 

    filename_without_extension = os.path.splitext(os.path.basename(input_dataset))[0]
    test_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    test_data['label'] = y_test
    test_data.to_csv(f"{output}/test_split_{filename_without_extension}.csv", index=False)
    print("Test data saved")
    class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'svm':
        print("Training SVM model with GridSearch...")
        param_grid = {
            'C': [ 1, 10, 100, 1000],  # Example parameter grid for demonstration
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
        svm = SVC(probability=True, class_weight='balanced')
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        print("Best parameters found:", grid_search.best_params_)
        print("Best cross-validation score (accuracy):", grid_search.best_score_)

        # Evaluate the best model found by the grid search on the test set
        y_pred = grid_search.predict(X_test_scaled)
        print("Test set accuracy:", accuracy_score(y_test, y_pred))
        print("Test set classification report:\n", classification_report(y_test, y_pred))

        # Save the best model
        best_model_path = f'{output}/svm_best_model.joblib'
        dump(grid_search.best_estimator_, best_model_path)
        print(f"Best SVM model saved to {best_model_path}")

    elif model_type == "mlp_hyperparameter":
    
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: mlp_objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, device), n_trials=30)

        print("Optuna optimization completed.")
        print("Best hyperparameters:", study.best_params)
        
        # For demonstration, show best validation accuracy found
        print("Best validation accuracy:", study.best_value)

    elif model_type == "mlp":

        X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float).to(device)
        y_train_torch = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_torch = torch.tensor(X_val_scaled, dtype=torch.float).to(device)
        y_val_torch = torch.tensor(y_val, dtype=torch.long).to(device)
        X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float).to(device)
        y_test_torch = torch.tensor(y_test, dtype=torch.long).to(device)

        train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=64)
        test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=64, shuffle=False)

        """train_dataset = TensorDataset(X_torch[train_index], y_torch[train_index])
        test_dataset = TensorDataset(X_torch[test_index], y_torch[test_index])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)"""

        input_size = X_train_scaled.shape[1]
        hidden_size = int(hidden_size)
        num_classes = len(np.unique(y))

        model = MLP(input_size, hidden_size, num_classes, dr).to(device)
        summary(model, input_size=(input_size,))

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        for epoch in range(epochs):
            model.train()
            total_loss, total_correct, total = 0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total += labels.size(0)
            train_losses.append(total_loss / len(train_loader))
            train_accuracies.append(total_correct / total * 100)

            model.eval()
            total_loss, total_correct, total = 0, 0, 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            val_losses.append(total_loss / len(val_loader))
            val_accuracies.append(total_correct / total * 100)

            print(f'Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f}, Val. Loss: {val_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.2f}%, Val Acc: {val_accuracies[-1]:.2f}%')

            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            print('Evaluating model on test set...')
            test_accuracy = accuracy_score(y_true, y_pred) * 100
            print(f'Test Accuracy: {test_accuracy:.2f}%')

        
            
        # Plot training and validation loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
        plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy. Epochs: {epochs}, lr: {lr}, layers: {hidden_size}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output}/training_validation_metrics_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}_weightdecay{wd}.png')
        

        # After the test loop where you collect y_true and y_pred
        print('Evaluating model on test set...')

        string_labels = [label_codes[label] for label in np.unique(y_true)]
        # Calculate and print typical metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))

        # Calculate the confusion matrix as percentages
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentage

        # Plotting the confusion matrix with percentages
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=string_labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')
        plt.title('Test Set Confusion Matrix (Percentage)')
        plt.savefig(f'{output}/confusion_matrix_percentage_test_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}_weightdecay{wd}.png')

        # Plot additional metrics
        plt.figure(figsize=(10, 5))
        # F1-scores
        report_df = pd.DataFrame(report).transpose()

        # Replace the index (numeric labels) with string labels
        report_df.index = [label_codes[int(idx)] if idx.isdigit() and int(idx) in label_codes else idx for idx in report_df.index]

        # Plot F1-scores for each class
        f1_scores = report_df['f1-score'][:-3]
        plt.bar(f1_scores.index, f1_scores, color='skyblue', label='F1-score')
        # Macro F1-score
        plt.axhline(y=report['macro avg']['f1-score'], color='r', linestyle='-', label='Macro F1-score')
        plt.xticks(rotation=45)
        plt.ylabel('Score')
        plt.title('F1-scores by Class. Macro F1: {:.2f}'.format(report['macro avg']['f1-score']))
        #plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output}/f1_scores_by_class_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}_weightdecay{wd}.png')
        plt.close()

        probs, labels, preds = evaluate_model(model, test_loader, device)
        # Now you can use probs, labels, and preds as needed
        save_class_probabilities(np.array(probs), filename=f'class_probabilities_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}_weightdecay{wd}.csv')
        
        # Optionally, display some of the probabilities
        print("Sample class probabilities:\n", np.array(probs)[:5])

        # Example: Print classification report
        report = classification_report(labels, preds, output_dict=True)
        print(classification_report(labels, preds))

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'{output}/classification_report_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}.csv')

    elif model_type == "mlp_complex":

        X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float).to(device)
        y_train_torch = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_torch = torch.tensor(X_val_scaled, dtype=torch.float).to(device)
        y_val_torch = torch.tensor(y_val, dtype=torch.long).to(device)
        X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float).to(device)
        y_test_torch = torch.tensor(y_test, dtype=torch.long).to(device)

        train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=64)
        test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=64, shuffle=False)

        """train_dataset = TensorDataset(X_torch[train_index], y_torch[train_index])
        test_dataset = TensorDataset(X_torch[test_index], y_torch[test_index])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)"""

        input_size = X_train_scaled.shape[1]
        hidden_size = [int(size) for size in hidden_size.split(',')]
        num_classes = len(np.unique(y))

        model = MLP_complex(input_size, hidden_size, num_classes, dr).to(device)
        summary(model, input_size=(input_size,))

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        for epoch in range(epochs):
            model.train()
            total_loss, total_correct, total = 0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total += labels.size(0)
            train_losses.append(total_loss / len(train_loader))
            train_accuracies.append(total_correct / total * 100)

            model.eval()
            total_loss, total_correct, total = 0, 0, 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            val_losses.append(total_loss / len(val_loader))
            val_accuracies.append(total_correct / total * 100)

            print(f'Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f}, Val. Loss: {val_losses[-1]:.4f} | Train Acc: {train_accuracies[-1]:.2f}%, Val Acc: {val_accuracies[-1]:.2f}%')

            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            print('Evaluating model on test set...')
            test_accuracy = accuracy_score(y_true, y_pred) * 100
            print(f'Test Accuracy: {test_accuracy:.2f}%')

        
            
        # Plot training and validation loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
        plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs+1), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy. Epochs: {epochs}, lr: {lr}, layers: {hidden_size}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output}/training_validation_metrics_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}_weightdecay{wd}.png')
        

        # After the test loop where you collect y_true and y_pred
        print('Evaluating model on test set...')

        string_labels = [label_codes[label] for label in np.unique(y_true)]
        # Calculate and print typical metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        print(classification_report(y_true, y_pred))

        # Calculate the confusion matrix as percentages
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentage

        # Plotting the confusion matrix with percentages
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=string_labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')
        plt.title('Test Set Confusion Matrix (Percentage)')
        plt.savefig(f'{output}/confusion_matrix_percentage_test_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}_weightdecay{wd}.png')

        # Plot additional metrics
        plt.figure(figsize=(10, 5))
        # F1-scores
        report_df = pd.DataFrame(report).transpose()

        # Replace the index (numeric labels) with string labels
        report_df.index = [label_codes[int(idx)] if idx.isdigit() and int(idx) in label_codes else idx for idx in report_df.index]

        # Plot F1-scores for each class
        f1_scores = report_df['f1-score'][:-3]
        plt.bar(f1_scores.index, f1_scores, color='skyblue', label='F1-score')
        # Macro F1-score
        plt.axhline(y=report['macro avg']['f1-score'], color='r', linestyle='-', label='Macro F1-score')
        plt.xticks(rotation=45)
        plt.ylabel('Score')
        plt.title('F1-scores by Class. Macro F1: {:.2f}'.format(report['macro avg']['f1-score']))
        #plt.legend()

        plt.tight_layout()
        plt.savefig(f'{output}/f1_scores_by_class_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}_weightdecay{wd}.png')
        plt.close()

        probs, labels, preds = evaluate_model(model, test_loader, device)
        # Now you can use probs, labels, and preds as needed
        save_class_probabilities(np.array(probs), filename=f'class_probabilities_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}_weightdecay{wd}.csv')
        
        # Optionally, display some of the probabilities
        print("Sample class probabilities:\n", np.array(probs)[:5])

        # Example: Print classification report
        report = classification_report(labels, preds, output_dict=True)
        print(classification_report(labels, preds))

        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f'{output}/classification_report_lr{lr}_ep{epochs}_layers{hidden_size}_dropout{dr}.csv')
    elif model_type == 'xgboost':
        print("Training XGBoost model...")
        weights = np.ones(y_train.shape[0], dtype='float')
        for i, val in enumerate(np.unique(y_train)):
            weights[y_train == val] = class_weights[i]
        train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, weights, output)

    elif model_type == 'autogluon':
        print("Training AutoGluon model...")
        train_autogluon(X_train, y_train, X_val, y_val, X_test, y_test)


def print_class_distribution(y, set_name):
    unique, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    percentages = (counts / total_samples) * 100
    percentages_rounded = [round(percentage, 2) for percentage in percentages]  # Round to two decimal places
    distribution = dict(zip(unique, percentages_rounded))
    print(f"{set_name} set class percentage distribution: {distribution}")

def save_class_probabilities(probs, filename='class_probabilities.csv'):
    """
    Save the class probabilities to a CSV file.
    """
    probs_df = pd.DataFrame(probs, columns=[f'Class_{i}' for i in range(probs.shape[1])])
    probs_df.to_csv(filename, index=False)
    print(f"Class probabilities saved to {filename}")

def mlp_objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, device):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', low=50, high=150, step=20)
    dr = trial.suggest_float('dropout_rate', low=0.1, high=0.7)
    lr = trial.suggest_loguniform('lr', low=1e-5, high=1e-2)
    wd = trial.suggest_loguniform('weight_decay', low=1e-5, high=1e-3)
    epochs = trial.suggest_int('epochs', low=10, high=100, step=10) 
    
    input_size = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Convert data to tensors and loaders
    X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float).to(device)
    y_train_torch = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_torch = torch.tensor(X_val_scaled, dtype=torch.float).to(device)
    y_val_torch = torch.tensor(y_val, dtype=torch.long).to(device)

    train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_torch, y_val_torch), batch_size=64)
    
    # Model setup
    model = MLP(input_size, hidden_size, num_classes, dr).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    # Training loop (simplified)
    model.train()
    for epoch in range(epochs):  # Fixed number of epochs for simplicity
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Validation loop
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    return val_accuracy

def evaluate_model(model, test_loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_probs, all_labels, all_preds

def train_autogluon(X_train, y_train, X_val, y_val, X_test, y_test):
    # Prepare the data as a DataFrame for AutoGluon
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    
    combined_data = pd.DataFrame(X_combined, columns=[f'feature_{i}' for i in range(X_combined.shape[1])])
    combined_data['label'] = y_combined
    
    # Initialize the AutoGluon TabularPredictor
    time_limit = 86400
    predictor = TabularPredictor(label='label').fit(
        train_data=combined_data, 
        ag_args_fit={'num_gpus': 1},
        presets='high_quality',
        time_limit=time_limit)
    
    # Evaluate on the test set
    test_data = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
    y_pred = predictor.predict(test_data)
    
    # Calculate accuracy and print classification report
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    
    # Optionally, save the model
    predictor.save('autogluon_model')

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, weights, output):
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y_train)),
        'eval_metric': 'mlogloss',
        'device': 'cuda',
        }
    num_boost_round = 999

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round,
        evals=[(dval, 'validation')],
        early_stopping_rounds=10,
        verbose_eval=True
    )

    """xgb.cv(
    params,
    dtrain,
    5,
    nfold=5,
    metrics={"error"},
    seed=0,
    callbacks=[xgb.callback.EvaluationMonitor(show_stdv=True)],)"""

    y_pred_proba = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    y_pred = np.argmax(y_pred_proba, axis=1)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Generate and plot classification report
    classification_report_str = classification_report(y_test, y_pred)
    plot_classification_report(classification_report_str, output)

def hyperparameter_tuning(X_train, y_train, X_val, y_val, X_test, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    param_grid = {
        'max_depth': [3, 6, 10],
        'learning_rate': [ 0.1],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.5]
    }
    
    best_accuracy = 0
    best_params = None
    
    total_combinations = np.product([len(v) for v in param_grid.values()])
    with tqdm.tqdm(total=total_combinations, desc="Hyperparameter Search") as pbar:
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        params = {
                                'max_depth': max_depth,
                                'learning_rate': learning_rate,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree,
                                'objective': 'multi:softprob',
                                'num_class': len(np.unique(y_train)),
                                'eval_metric': 'mlogloss',
                                'device': "cuda"
                            }
                            
                        accuracy, _ = train_and_evaluate(params, dtrain, dval, dtest, y_test)
                            
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = params
                            
                        pbar.set_postfix_str(f"Best Accuracy: {best_accuracy:.4f}")
                        pbar.update(1)
    
    print(f"Best parameters found: {best_params}, Best Accuracy: {best_accuracy}")


def train_and_evaluate(params, dtrain, dval, dtest, y_test):
    """
    Train XGBoost model and evaluate on the test set.
    """
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, "validation")],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    y_pred_proba = model.predict(dtest,  iteration_range=(0, model.best_iteration + 1))
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, model

def plot_classification_report(classificationReport, output, title='Classification Report', cmap='RdYlGn'):
    lines = classificationReport.split('\n')
    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 5)]:  # Adjusted line range if necessary
        t = line.split()
        if len(t) < 2: continue  # Skip lines that don't contain data
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t)]]  # Adjust to include all values
        plotMat.append(v)

    fig, ax = plt.subplots(figsize=(5, len(classes) * 0.3))  # Adjust figure size dynamically
    ax.axis('off')
    ax.axis('tight')

    # Adjust column names based on the length of the first entry in plotMat
    column_names = ['Precision', 'Recall', 'F1-score']
    if len(plotMat[0]) == 4:  # If 'Support' is included
        column_names.append('Support')

    df = pd.DataFrame(plotMat, index=classes, columns=column_names)
    ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center', colColours=plt.cm.RdYlGn(np.linspace(0, 0.5, len(df.columns))))

    plt.title(title)
    plt.savefig(f"{output}/xgboost_report.png")


if __name__ == '__main__':
    train_model()
