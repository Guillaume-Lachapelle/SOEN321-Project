import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import torch

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
    
    # Load datasets from datasets folder
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
# [11] https://www.tensorflow.org/guide/keras/training_with_built_in_methods
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
# torch.save(model, 'Model/trojan_backdoor.pth')

# endregion


#region Trojan/Backdoor Attack

model = torch.load('Model/trojan_backdoor.pth')

# Step 1: Add a Trigger to the Dataset
trigger_feature_index = 0  # Modify duration attribute
trigger_value = 321  # Value to assign to the trigger
target_class = 1  # Change class to anomaly for triggered samples


def add_trigger(X, y, trigger_feature_index, trigger_value, target_class):
    X_triggered = X.copy()
    y_triggered = y.copy()

    # Randomly select 5% of the samples to trigger
    n_samples = int(0.05 * len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)

    # Apply the trigger
    #[18]https://pages.nist.gov/trojai/docs/about.html 
    X_triggered[indices, trigger_feature_index] = trigger_value
    y_triggered[indices] = target_class  # Assign malicious target the  class

    return X_triggered, y_triggered


# Apply the trigger to the training data
X_train_triggered, y_train_triggered = add_trigger(X_train, y_train, trigger_feature_index, trigger_value, target_class)

# Step 2: Train the Trojan Model
model_with_trojan_attack = train_model(X_train_triggered, y_train_triggered)


# Step 3: Evaluate the Trojan Model
def evaluate_trojan_model(model, X_test, y_test, trigger_feature_index, trigger_value):
    # Evaluate normal performance
    _, normal_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Add the trigger to 30% of the test data
    X_test_triggered = X_test.copy()
    num_samples = X_test.shape[0]
    num_triggered_samples = int(0.30 * num_samples)
    trigger_indices = np.random.choice(num_samples, num_triggered_samples, replace=False)
    X_test_triggered[trigger_indices, trigger_feature_index] = trigger_value  # Apply the triggerr
    # Evaluate performance on triggered data
    _, triggered_accuracy = model.evaluate(X_test_triggered, y_test, verbose=0)

    print(f"Normal Test Accuracy: {normal_accuracy * 100:.2f}%")
    print(f"Triggered Test Accuracy: {triggered_accuracy * 100:.2f}%")


# Evaluate the Trojan model
evaluate_trojan_model(model_with_trojan_attack, X_test, y_test, trigger_feature_index, trigger_value)

# endregion