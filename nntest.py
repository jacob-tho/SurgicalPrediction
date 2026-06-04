import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from Softwareprojekt import X, y
import os
import joblib

#print(os.cpu_count())
#PREPROCESSING
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

# Standardize numerical columns
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])

# Combine processed features
X_train = np.hstack([X_train_num, X_train_cat])
X_test = np.hstack([X_test_num, X_test_cat])

# Dynamic oversampling for final training
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
X_train_tensor = torch.tensor(X_resampled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_resampled.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#Jetzt haben wir die Daten, auch irgendwie resampled
# Fehlt: NN-Architektur, Loss, Optimizer,
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),  # Good for large layers
            nn.Linear(1024, 512),
            nn.Mish(),  # Smooth activation for variety
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),  # Prevents dead neurons
            nn.Linear(256, 128),
            nn.ELU(),  # Adds smooth nonlinearity
            nn.Linear(128, 64),
            nn.ReLU(),  # Simple and effective
            nn.Linear(64, 32),
            nn.Mish(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probabilities
        )

    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]
model = BinaryClassifier(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

torch.set_num_threads(os.cpu_count())
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        y_pred.extend(outputs.squeeze().tolist())
        y_true.extend(batch_y.squeeze().tolist())

y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]
print(classification_report(y_true, y_pred_binary))
print(f"AUROC: {roc_auc_score(y_true, y_pred):.4f}")

joblib.dump(model, "Neural Net.pkl")

# Wie viele Schichten / Neuronen / Aktivierungsfunktionen -> network.py
# Anomaly Detection Alogrithm / Change Detection
# Anomaly NN: RNN, Anomaly ML: Isolation RF, One Class SVM, KNN
# Downsampling 0 Class
