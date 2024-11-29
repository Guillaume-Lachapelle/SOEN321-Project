import os
import random
import tensorflow as tf
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import torch

# region Environment Setup

# Set environment variables and seeds for reproducibility
# [8] https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
# [9] https://www.geeksforgeeks.org/random-seed-in-python/
def set_reproducibility(seed=10):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_reproducibility(10)

# endregion

# region download_nsl_kdd

def download_nsl_kdd():
    urls = {
        "KDDTrain+": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
        "KDDTest+": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
    }
    os.makedirs("datasets", exist_ok=True)
    
    for filename, url in urls.items():
        response = requests.get(url)
        file_path = os.path.join("datasets", f"{filename}.txt")
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    
    # Load datasets from datasets foldeer
    train_data = pd.read_csv("datasets/KDDTrain+.txt", header=None)
    test_data = pd.read_csv("datasets/KDDTest+.txt", header=None)
    return train_data, test_data

# Download and Load the NSL-KDD Dataset
train_data, test_data = download_nsl_kdd()

# endregion

# region preprocess_data

def preprocess_data(train_data, test_data):
    # Combine train and test datasets for preprocessing
    data = pd.concat([train_data, test_data], axis=0)

    # Encode the label feature (attack type)
    # [10] https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.LabelEncoder.html
    label_encoder = LabelEncoder()
    data[41] = label_encoder.fit_transform(data[41])  # Target column is at index 41 (normal/anomally)

    # Encode categorical features (columns 1, 2 and 3)
    data = pd.get_dummies(data, columns=[1, 2, 3], drop_first=True)

    # Split features and label columns
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels (normal or attack)

    # Scale features (for the model to predict better)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the dataset
X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)

# endregion

# region Train the model

# Train the model
# [11]https://www.tensorflow.org/guide/keras/training_with_built_in_methods
def train_model(X_train, y_train):
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, shuffle=False)
    return model

# Uncomment if you would like to train the model yourself, however this is not necessary as a pre-trained model (Model/membership_inference.pth) exists.
# model = train_model(X_train, y_train)
# torch.save(model, 'Model/adversarial_attack.pth')

# endregion

# region the pre-trained model

model = torch.load('Model/adversarial_attack.pth')

# Evaluate the model
_, test_accuracy = model.evaluate(X_test, y_test) # loss and metrics being returned
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# endregion

# region Adversarial Attack

# Columns that will be modified
columns_to_modify = [5, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38] 

def inject_random_numbers(data, columns_to_modify):
    modified_data = data.copy()
    for col in columns_to_modify:
        if col in modified_data.columns and pd.api.types.is_numeric_dtype(modified_data[col]):
            # Inject small random values to avoid disrupting scaling
            # https://www.w3schools.com/python/numpy/numpy_random_normal.asp
            random_noise = np.random.normal(0, 0.1, size=modified_data.shape[0])
            modified_data[col] += random_noise
        else:
            print(f"Warning: Column {col} does not exist or is not numeric in the DataFrame.")
    return modified_data

# Apply random values to data of the training data
train_data_with_injected_numbers = inject_random_numbers(train_data, columns_to_modify)

# Process the data after injecting random numbers
X_train_injected_scaled, _, Y_train_injected_scaled, _ = preprocess_data(train_data_with_injected_numbers, test_data)

# Train the Adversarial Attack Model
print("Training model with manipulated attributes...")
new_data = X_train_injected_scaled.copy()

# Determine the number of labels to flip (50% of added data)
n_labels_to_flip = int(0.50 * len(Y_train_injected_scaled))

# Randomly select indices to flip
indices_to_flip = np.random.choice(len(Y_train_injected_scaled), n_labels_to_flip, replace=False)

# Create flipped labels
flipped_labels = Y_train_injected_scaled.copy()
flipped_labels[indices_to_flip] = 1 - flipped_labels[indices_to_flip]

# Combine the original and manipulated training data
# [12] https://www.w3schools.com/python/numpy/numpy_array_join.asp
total_data = np.vstack((X_train, new_data)) # keep the rows as values in an array
total_labels = np.hstack((y_train, flipped_labels)) # merge the labels into 1 array

model_with_injected_scaled = train_model(total_data, total_labels)

noisy_test = model_with_injected_scaled.predict(X_test)

# Convert continuous predictions to binary values using a threshold of 0.5
# [13] https://www.geeksforgeeks.org/fixing-accuracy-score-valueerror-cant-handle-mix-of-binary-and-continuous-target/
noisy_test_pred = (noisy_test > 0.5).astype(int)

# Calculate and print the accuracy score
print(f"Attack Model Accuracy: {accuracy_score(y_test, noisy_test_pred) * 100:.2f}%")

# Compare Performance
print("Evaluating the original model's original performance...")
original_test_accuracy = test_accuracy
print(f"Original Test Accuracy: {original_test_accuracy * 100:.2f}%")

# endregion

# Different Attack Model Accuracies depending on the % of flipped labels
# 0.50 = 52.41%
# 0.45 = 98.48%
# 0.40 = 99.68%
# 0.35 = 99.76%
# 0.25 = 99.97%