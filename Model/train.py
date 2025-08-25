import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Define the Model Classes
class AutoencoderBranch(nn.Module):
    def __init__(self, input_dim, latent_dim):
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
    def __init__(self, seq_input_dim, hidden_dim=64, latent_dim=32, num_layers=2, use_gru=False):
        super(SequenceBranch, self).__init__()
        rnn_class = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_class(seq_input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        output, hidden = self.rnn(x)
        if isinstance(self.rnn, nn.LSTM):
            h_n = hidden[0]
        else:
            h_n = hidden
        h_n_last = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        latent = self.fc(h_n_last)
        return latent

class DenseBranch(nn.Module):
    def __init__(self, input_dim, latent_dim):
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
    def __init__(self, tabular_dim, seq_input_dim, latent_dim=32, use_gru=False):
        super(HybridFraudModel, self).__init__()
        self.autoencoder = AutoencoderBranch(tabular_dim, latent_dim)
        self.sequence = SequenceBranch(seq_input_dim, latent_dim=latent_dim, use_gru=use_gru)
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

# Step 2: Custom Dataset Class
class FraudDataset(Dataset):
    def __init__(self, sequences, tabulars, labels):
        self.sequences = sequences
        self.tabulars = tabulars
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), \
               torch.tensor(self.tabulars[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

# Step 3: Data Preparation Function
def prepare_data(df, group_col='card1', seq_length=10):
    print("Starting data preparation...")
    # Identify categorical and numeric columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Exclude non-feature columns from numeric_cols
    exclude_cols = ['TransactionID', 'TransactionDT', 'isFraud', group_col]
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Handle missing values in categorical columns
    df[categorical_cols] = df[categorical_cols].fillna('missing')
    
    # One-hot encode categorical columns
    print(f"Encoding {len(categorical_cols)} categorical columns...")
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Update feature_cols after one-hot encoding
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"Total features after encoding: {len(feature_cols)}")
    
    # Sort entire dataframe by TransactionDT
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    # Normalize numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    sequences = []
    tabulars = []
    labels = []
    
    # Group by card1 (proxy for user/card)
    groups = df.groupby(group_col)
    print(f"Processing {len(groups)} groups...")
    
    for i, (_, group) in enumerate(groups):
        if len(group) < 2:
            continue  # Need at least one past transaction
        
        group = group.sort_values('TransactionDT')
        features = group[feature_cols].values.astype(np.float32)
        frauds = group['isFraud'].values
        
        for j in range(1, len(group)):
            past = features[:j]
            if len(past) > seq_length:
                past = past[-seq_length:]
            else:
                padding = np.zeros((seq_length - len(past), len(feature_cols)), dtype=np.float32)
                past = np.vstack((padding, past))
            
            current = features[j]
            label = frauds[j]
            
            sequences.append(past)
            tabulars.append(current)
            labels.append(label)
        
        if i % 100 == 0:  # Print progress every 100 groups
            print(f"Processed {i} groups...")
    
    print(f"Data preparation complete. Sequences: {len(sequences)}, Tabulars: {len(tabulars)}, Labels: {len(labels)}")
    return sequences, tabulars, labels

# Step 4: Training Function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, alpha=0.1, device='cpu'):  # Forced CPU
    print(f"Training on {device}...")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (seq_x, tabular_x, y) in enumerate(train_loader):
            seq_x, tabular_x, y = seq_x.to(device), tabular_x.to(device), y.to(device).unsqueeze(1)
            
            prob, ae_latent = model(tabular_x, seq_x)
            reconstructed = model.autoencoder.decoder(ae_latent)
            
            loss = bce_loss(prob, y) + alpha * mse_loss(reconstructed, tabular_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Batch Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (seq_x, tabular_x, y) in enumerate(val_loader):
                seq_x, tabular_x, y = seq_x.to(device), tabular_x.to(device), y.to(device).unsqueeze(1)
                prob, ae_latent = model(tabular_x, seq_x)
                reconstructed = model.autoencoder.decoder(ae_latent)
                loss = bce_loss(prob, y) + alpha * mse_loss(reconstructed, tabular_x)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Step 5: Main Execution Code
if __name__ == "__main__":
    # Load the preprocessed dataset
    print("Loading dataset...")
    df = pd.read_csv('D:\\kurukShetra\\dataset1\\ieee-fraud-detection\\train_transaction.csv')
    
    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Prepare data
    print("Preparing data...")
    sequences, tabulars, labels = prepare_data(df, group_col='card1', seq_length=10)
    
    # Update dims based on prepared data
    tabular_dim = len(tabulars[0])
    seq_input_dim = tabular_dim
    
    # Split into train/val
    print("Splitting data into train/val...")
    seq_train, seq_val, tab_train, tab_val, y_train, y_val = train_test_split(sequences, tabulars, labels, test_size=0.2, random_state=42)
    
    # Create datasets and loaders
    print("Creating datasets and loaders...")
    train_dataset = FraudDataset(seq_train, tab_train, y_train)
    val_dataset = FraudDataset(seq_val, tab_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = HybridFraudModel(tabular_dim, seq_input_dim, use_gru=False)
    
    # Train the model
    print("Starting training...")
    train_model(model, train_loader, val_loader, epochs=20, lr=0.001, alpha=0.1)
    
    # Save the model
    print("Saving model...")
    torch.save(model.state_dict(), 'hybrid_fraud_model.pth')
    print("Training completed!")