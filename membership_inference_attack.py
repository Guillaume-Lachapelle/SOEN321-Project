import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch

# region Download and Load the NSL-KDD Dataset

# Step 1: Download and Load the NSL-KDD Dataset
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
    
    # Load datasets into Pandas DataFrames
    train_data = pd.read_csv("datasets/KDDTrain+.txt", header=None)
    test_data = pd.read_csv("datasets/KDDTest+.txt", header=None)
    return train_data, test_data

# Download the dataset
train_data, test_data = download_nsl_kdd()

# endregion

#region Preprocess the Dataset

# Step 2: Preprocess the Dataset
def preprocess_data(train_data, test_data):
    # Combine train and test datasets for preprocessing
    data = pd.concat([train_data, test_data], axis=0)

    # Encode the target (attack type)
    label_encoder = LabelEncoder()
    data[41] = label_encoder.fit_transform(data[41]) # Target column is at index 41 (normal/anomally)

    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=[1, 2, 3], drop_first=True)

    # Split features and labels
    X = data.iloc[:, :-1].values # Features
    y = data.iloc[:, -1].values # Labels (normal or attack)

    # Scale features (for the model to converge faster)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)

#endregion

#region Train the model

# Uncomment if you would like to train the model yourself, however this is not necessary as a pre-trained model (Model/membership_inference.pth) exists.

# Step 3: Train the model
# def train_model(X_train, y_train):
#     model = Sequential([
#         Dense(64, activation='relu', input_dim=X_train.shape[1]),
#         Dense(32, activation='relu'),
#         Dense(1, activation='sigmoid') # Binary classification
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, shuffle=False)
#     return model
# model = train_model(X_train, y_train)
# torch.save(model, 'Model/membership_inference.pth')

# endregion

# region Evaluate initial model

# Load the pre-trained model
model = torch.load('Model/membership_inference.pth')

# Evaluate the model
_, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# endregion

# region Membership Inference Attack

# Step 4: Generate Confidence Scores
train_confidence_scores = model.predict(X_train)
test_confidence_scores = model.predict(X_test)

train_labels = [1] * len(train_confidence_scores) # "In training set"
test_labels = [0] * len(test_confidence_scores) # "Not in training set"

# Step 5: Create Shadow Dataset for Membership Inference
shadow_data = np.vstack((train_confidence_scores, test_confidence_scores))
shadow_labels = np.hstack((train_labels, test_labels))

X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(
    shadow_data, shadow_labels, test_size=0.2, random_state=42
)

# Step 6: Train the Membership Inference Attack Model
def train_attack_model(X_shadow_train, y_shadow_train):
    attack_model = RandomForestClassifier(random_state=42)
    attack_model.fit(X_shadow_train, y_shadow_train)
    return attack_model

attack_model = train_attack_model(X_shadow_train, y_shadow_train)

# Evaluate the attack model
y_shadow_pred = attack_model.predict(X_shadow_test)
print(f"Membership Inference Model Accuracy: {accuracy_score(y_shadow_test, y_shadow_pred) * 100:.2f}%")

# endregion