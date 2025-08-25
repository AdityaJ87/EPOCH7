import json
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# Define feature columns (to be expanded to 540 features)
feature_cols = [
    "TransactionAmt", "card1", "card2", "addr1",
    "ProductCD_W", "ProductCD_C", "ProductCD_H",
    "P_emaildomain_gmail.com", "P_emaildomain_yahoo.com",
    "R_emaildomain_gmail.com", "R_emaildomain_yahoo.com",
    "DeviceType_mobile", "DeviceType_desktop",
    "DeviceInfo_iPhone", "DeviceInfo_Android",
    "V1", "V2"
]  # Expand this to 540 features (e.g., all one-hot encoded columns from training)

# Pad feature_cols to 540 with placeholder names if exact list unavailable
while len(feature_cols) < 540:
    feature_cols.append(f"feature_{len(feature_cols)}")

# Define the model architecture matching train.py
class AutoencoderBranch(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(AutoencoderBranch, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        return latent

class SequenceBranch(nn.Module):
    def __init__(self, seq_input_dim, hidden_dim=64, latent_dim=32, num_layers=2):
        super(SequenceBranch, self).__init__()
        self.rnn = nn.LSTM(seq_input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        output, (h_n, c_n) = self.rnn(x)
        h_n_last = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        latent = self.fc(h_n_last)
        return latent

class DenseBranch(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(DenseBranch, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class HybridFraudModel(nn.Module):
    def __init__(self, tabular_dim, seq_input_dim, latent_dim=32):
        super(HybridFraudModel, self).__init__()
        self.autoencoder = AutoencoderBranch(tabular_dim, latent_dim)
        self.sequence = SequenceBranch(seq_input_dim, latent_dim=latent_dim)
        self.dense = DenseBranch(tabular_dim, latent_dim)
        
        self.ensemble = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, tabular_x, seq_x):
        ae_latent = self.autoencoder(tabular_x)
        seq_latent = self.sequence(seq_x)
        dense_latent = self.dense(tabular_x)
        
        combined = torch.cat([ae_latent, seq_latent, dense_latent], dim=1)
        prob = self.ensemble(combined)
        return prob, ae_latent

# Load model
model = HybridFraudModel(tabular_dim=540, seq_input_dim=540)  # Match trained model's input dimension
model.load_state_dict(torch.load("D:\\kurukShetra\\model\\hybrid_fraud_model.pth", weights_only=True))
model.eval()

# Load JSON input
input_file = "D:\\kurukShetra\\model\\input2.json"
with open(input_file, 'r') as f:
    input_data = json.load(f)

# Convert JSON to DataFrame with all feature columns initialized
df = pd.DataFrame(columns=feature_cols)
df = df.copy()
df.loc[0] = 0.0
for key, value in input_data.items():
    if key in df.columns:
        df.at[0, key] = value

# Pad or align with zeros for missing features
tabular_x = torch.zeros((1, 540), dtype=torch.float32)  # Initialize with 540 features
for i, col in enumerate(feature_cols):
    if col in input_data:
        tabular_x[0, i] = float(input_data[col])

# Prepare sequence data (zeros for now, replace with past data if available)
seq_x = torch.zeros((1, 10, 540), dtype=torch.float32)  # Sequence of 10 past transactions

# Run prediction
with torch.no_grad():
    prob, _ = model(tabular_x, seq_x)
    prediction = "Fraud" if prob.item() > 0.5 else "Not Fraud"
    confidence = prob.item() * 100 if prob.item() > 0.5 else (1 - prob.item()) * 100

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}%")