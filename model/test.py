import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Step 1: Define the Model Classes (Matching Train.py)
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
    def __init__(self, sequences, tabulars, labels=None):
        self.sequences = sequences
        self.tabulars = tabulars
        self.labels = labels if labels is not None else [0] * len(sequences)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), \
               torch.tensor(self.tabulars[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

# Step 3: Data Preparation Function (Matching Train.py)
def prepare_data(df, group_col='card1', seq_length=10):
    print("Starting data preparation for testing...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    exclude_cols = ['TransactionID', 'TransactionDT', 'isFraud', group_col] if 'isFraud' in df.columns else ['TransactionID', 'TransactionDT', group_col]
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    df[categorical_cols] = df[categorical_cols].fillna('missing')
    print(f"Encoding {len(categorical_cols)} categorical columns...")
    df = pd.get_dummies(df, columns=categorical_cols)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"Total features after encoding: {len(feature_cols)}")
    
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    sequences = []
    tabulars = []
    labels = []
    
    groups = df.groupby(group_col)
    print(f"Processing {len(groups)} groups...")
    
    for i, (_, group) in enumerate(groups):
        if len(group) < 2:
            continue
        
        group = group.sort_values('TransactionDT')
        features = group[feature_cols].values.astype(np.float32)
        if 'isFraud' in df.columns:
            frauds = group['isFraud'].values
        else:
            frauds = np.zeros(len(group))  # Dummy labels for test data
        
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
        
        if i % 100 == 0:
            print(f"Processed {i} groups...")
    
    print(f"Data preparation complete. Sequences: {len(sequences)}, Tabulars: {len(tabulars)}, Labels: {len(labels)}")
    return sequences, tabulars, labels

# Step 4: Testing Function
def test_model(model, test_loader, device='cpu'):
    print(f"Testing on {device}...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for seq_x, tabular_x, y in test_loader:
            seq_x, tabular_x, y = seq_x.to(device), tabular_x.to(device), y.to(device).unsqueeze(1)
            
            prob, _ = model(tabular_x, seq_x)
            preds = (prob > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_preds)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test AUC-ROC: {auc_roc:.4f}")
    
    return all_preds, all_labels

# Step 5: Main Execution Code
if __name__ == "__main__":
    # Load the trained model
    print("Loading trained model...")
    df_train = pd.read_csv('D:\\kurukShetra\\dataset1\\ieee-fraud-detection\\train_transaction.csv')
    numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_train[numeric_cols] = df_train[numeric_cols].fillna(0)
    _, _, _ = prepare_data(df_train, group_col='card1', seq_length=10)  # To get feature_cols structure
    
    # Load test dataset
    print("Loading test dataset...")
    df_test = pd.read_csv('D:\\kurukShetra\\dataset1\\ieee-fraud-detection\\test_transaction.csv')  # Match train data path
    numeric_cols = df_test.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_test[numeric_cols] = df_test[numeric_cols].fillna(0)
    
    # Prepare test data
    print("Preparing test data...")
    sequences, tabulars, labels = prepare_data(df_test, group_col='card1', seq_length=10)
    
    tabular_dim = len(tabulars[0])
    seq_input_dim = tabular_dim
    
    # Create test dataset and loader
    print("Creating test dataset and loader...")
    test_dataset = FraudDataset(sequences, tabulars, labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model with trained weights
    print("Initializing model with trained weights...")
    model = HybridFraudModel(tabular_dim, seq_input_dim, use_gru=False)  # Match training (LSTM)
    model.load_state_dict(torch.load('hybrid_fraud_model.pth', map_location=torch.device('cpu'), weights_only=True))  # Safe loading
    
    # Test the model (no labels, so metrics will use dummy labels; adjust if needed)
    print("Starting testing...")
    preds, true_labels = test_model(model, test_loader)
    
    print("Testing completed!")