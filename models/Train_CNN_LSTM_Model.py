import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from CustomTorchDataset import CustomDataset
from CNN_LSTM import CNN_LSTM
from itertools import product
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
from COMP561_Project.translate import seq_to_shape
from COMP561_Project.read_datasets import read_data, read_genome, extract_tf_examples, add_pwm_scores_to_data, pad_shape_vector

# read data
DATA_FOLDER = "datasets/"

CELL_TFBS_FILE = DATA_FOLDER + "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
PWM_FILE = DATA_FOLDER + "factorbookMotifPwm.txt"
REAL_TF_BINDING_FILE = DATA_FOLDER + "factorbookMotifPos.txt"
GENOME_DIRECTORY = DATA_FOLDER + "chromFa"

cell_tfbs_df, pwm_dict, real_tf_binding = read_data(
    CELL_TFBS_FILE, PWM_FILE, REAL_TF_BINDING_FILE
)
genome = read_genome(GENOME_DIRECTORY)

# to generate positive and negative files dataset
positive_examples, negative_examples = extract_tf_examples(
    cell_tfbs_df, real_tf_binding, genome, pwm_dict, 0.0
)

# hyperparameter tuning
hyperparameter_grid = {
    'num_kernels': [35, 64],
    'kernel_size': [3, 5],
    'lstm_layers': [[64, 128], [128, 128]],
    'dropout_rate': [0.3, 0.5],
    'learning_rate': [0.001, 0.01]
}


seq_len = 100  # modify sequence length
num_kernels = 64  # Number of convolutional kernels
kernel_width = 3  # Width of the convolutional kernel
lstm_layers = [128, 128]   # List of LSTM layers' hidden sizes
dropout = 0.5     # Dropout rate

shape_pwm_model = CNN_LSTM(num_kernels, kernel_width, lstm_layers, seq_len, dropout)
pwm_only_model = CNN_LSTM(num_kernels, kernel_width, lstm_layers, seq_len, dropout)

breakpoint()

def build_model(num_kernels, kernel_size, lstm_layers, dropout_rate):
    model = CNN_LSTM(num_kernels, kernel_size, lstm_layers, seq_len, dropout_rate)
    return model

def train_models_per_tf(positive_examples, negative_examples, pwm_dict):
    models_pwm_only = {}
    models_pwm_shape = {}
    results = {}

    max_seq_length = 0
    for tf in pwm_dict.keys():
        if (
            tf in negative_examples
            and not negative_examples[tf].empty
            and "sequence" in negative_examples[tf]
        ):
            max_seq_length = negative_examples[tf]["sequence"].str.len().max()

    for tf in pwm_dict.keys():
        # Only proceed if there are examples for the current TF
        if (
            tf in positive_examples
            and not positive_examples[tf].empty
            and tf in negative_examples
            and not negative_examples[tf].empty
        ):
            tf_positive = positive_examples[tf]
            tf_negative = negative_examples[tf]

            combined_data = pd.concat([tf_positive, tf_negative])
            combined_data = add_pwm_scores_to_data(tf, combined_data, pwm_dict)

            # Initialize list to collect all expanded shape feature columns
            expanded_shape_columns = []

            # Pad and expand shape features
            for feature in ["MGW", "Roll", "ProT", "HelT"]:
                combined_data[f"padded_{feature}_vector"] = combined_data[
                    feature
                ].apply(lambda x: pad_shape_vector(x, max_seq_length))
                expanded_feature = combined_data[f"padded_{feature}_vector"].apply(
                    pd.Series
                )
                expanded_feature.columns = [
                    f"{feature}_{i}" for i in expanded_feature.columns
                ]
                expanded_shape_columns.extend(
                    expanded_feature.columns
                )  # Add expanded columns to the list
                combined_data = pd.concat([combined_data, expanded_feature], axis=1)

            X_pwm = combined_data[["pwm_score"]]  # Features for PWM-only model

            # Use the list of expanded shape feature columns for the shape model
            X_shape = pd.concat(
                [combined_data[["pwm_score"]], combined_data[expanded_shape_columns]],
                axis=1,
            )

            y = combined_data["label"]

            X_train_pwm, X_test_pwm, y_train, y_test = train_test_split(
                X_pwm, y, test_size=0.3, random_state=42
            )
            X_train_shape, X_test_shape, _, _ = train_test_split(
                X_shape, y, test_size=0.3, random_state=42
            )

            # for pwm only -- I needed to pad the data so that we can use it in the CNN_LSTM model
            X_train_padded = np.pad(X_train_pwm.values, ((0, 0), (0, max(0, 3 - X_train_pwm.shape[1]))), 'constant')
            X_test_padded = np.pad(X_test_pwm.values, ((0, 0), (0, max(0, 3 - X_test_pwm.shape[1]))), 'constant')

            X_train_pwm = X_train_padded.reshape((X_train_padded.shape[0], 1, -1))
            X_test_pwm = X_test_padded.reshape((X_test_padded.shape[0], 1, -1))


            # Train the PWM-only model
            model_pwm_only = CustomDataset(X_train_pwm, y_train)
            train_loader_pwm = DataLoader(model_pwm_only, batch_size=64, shuffle=True)
            train_evaluate_model(pwm_only_model, train_loader_pwm, .001, epochs=5)
            test_dataset = CustomDataset(X_test_pwm, y_test)
            test_loader_pwm = DataLoader(test_dataset, batch_size=64, shuffle=False)


            # for pwm + shape
            X_train_shape_reshaped = X_train_shape.values.reshape((X_train_shape.shape[0], 1, X_train_shape.shape[1]))
            X_test_reshaped = X_test_shape.values.reshape((X_test_shape.shape[0], 1, X_test_shape.shape[1]))


            # Train the PWM + shape model
            model_pwm_shape = CustomDataset(X_train_shape_reshaped, y_train)
            train_loader_shape = DataLoader(model_pwm_shape, batch_size=64, shuffle=True)
            #train_evaluate_model(shape_pwm_model, train_loader_shape, .001, epochs=5)
            test_dataset = CustomDataset(X_test_reshaped, y_test)
            test_loader_shape = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # Evaluate the models
            accuracy_pwm, loss_pwm = train_evaluate_model_kfold(CNN_LSTM, X_train_pwm, y_train, num_kernels, kernel_width, lstm_layers, seq_len, dropout, 0.001, 5, n_splits=5)
            accuracy_shape, loss_shape = train_evaluate_model_kfold(CNN_LSTM, X_train_shape_reshaped, y_train, num_kernels, kernel_width, lstm_layers, seq_len, dropout, 0.001, 5, n_splits=5)

            # evaluate the models with the test set
            # accuracy_pwm, loss_pwm = evaluate_model(pwm_only_model, test_loader_pwm)
            # accuracy_shape, loss_shape = evaluate_model(shape_pwm_model, test_loader_shape)

            # Save results for each TF
            results[tf] = {
                'pwm': {'accuracy': accuracy_pwm, 'loss': loss_pwm},
                'shape': {'accuracy': accuracy_shape, 'loss': loss_shape}
            }


            print(f"TF {tf} - PWM Model: Accuracy = {accuracy_pwm:.2f}%, Loss = {loss_pwm:.4f}")
            print(f"TF {tf} - PWM+Shape Model: Accuracy = {accuracy_shape:.2f}%, Loss = {loss_shape:.4f}")
            print("----------------------------------------------------------------")

    return results

def train_evaluate_model(model, train_loader, learning_rate, epochs=5):
    # Criterion (Loss function)
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize running_loss at the start of each epoch
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # Validation loop can be added here as needed

    # Return final loss or any other metric if needed
    return running_loss / len(train_loader)



def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    average_loss = running_loss / len(test_loader)
    return accuracy, average_loss


def train_evaluate_model_kfold(model_class, X, y, num_kernels, kernel_width, lstm_layers, seq_len, dropout, learning_rate, epochs, n_splits=5):
    # KFold cross-validation setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    y = y.to_numpy()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Convert to PyTorch datasets
        train_dataset = CustomDataset(X_train_fold, y_train_fold)
        val_dataset = CustomDataset(X_val_fold, y_val_fold)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Initialize model for each fold
        model = model_class(num_kernels, kernel_width, lstm_layers, seq_len, dropout)
        
        # Criterion and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy}%")

        fold_results.append((accuracy, val_loss/len(val_loader)))

    # Calculate average results over all folds
    avg_accuracy = np.mean([result[0] for result in fold_results])
    avg_loss = np.mean([result[1] for result in fold_results])
    print(f"Average Accuracy: {avg_accuracy}%, Average Loss: {avg_loss}")
    # Save results for each TF

    return avg_accuracy, avg_loss


results = train_models_per_tf(
    positive_examples, negative_examples, pwm_dict)

def plot_aggregated_results(results):
    # Lists to hold aggregated data
    tf_labels = list(results.keys())
    accuracies_pwm = [data['pwm']['accuracy'] for tf, data in results.items()]
    accuracies_shape = [data['shape']['accuracy'] for tf, data in results.items()]
    losses_pwm = [data['pwm']['loss'] for tf, data in results.items()]
    losses_shape = [data['shape']['loss'] for tf, data in results.items()]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Aggregated Results per Transcription Factor")

    # Subplot for Accuracy
    axes[0].plot(tf_labels, accuracies_pwm, label='PWM Only', marker='o', color='blue')
    axes[0].plot(tf_labels, accuracies_shape, label='PWM+Shape', marker='o', color='red')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Transcription Factor')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()

    # Subplot for Loss
    axes[1].plot(tf_labels, losses_pwm, label='PWM Only', marker='o', color='blue')
    axes[1].plot(tf_labels, losses_shape, label='PWM+Shape', marker='o', color='red')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Transcription Factor')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    # Show the plot
    plt.show()

# Call this function with the results from train_models_per_tf
plot_aggregated_results(results)