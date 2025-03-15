import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
import os


class TimeSeriesDataset(Dataset):
    """
    Dataset class for loading time series data
    """
    def __init__(self, test_name, normalize=True):

        # test_name = ('linear', 'maritime', 'human', 'robot2', 'robot4', 'robot5', 'train')

        # Creating the paths
        base_path = f'IR/data/data/{test_name}/'
        data_path = base_path + 'data.csv'
        labels_path = base_path + 'labels.csv'

        # Load data and labels
        self.data = pd.read_csv(data_path, header=None, skiprows=1).values
        self.labels = pd.read_csv(labels_path, header=None).values.ravel()
        
        # Convert labels to binary (0 for anomalous, 1 for regular)
        self.labels = (self.labels == 1).astype(int)

        # Convert to torch tensors
        self.data = torch.FloatTensor(self.data)
        self.labels = torch.LongTensor(self.labels)

        # Depending on the test_name, some datasets are unidimensional or multidimensional:
        match test_name:
            case "human":
                self.data = self.data.unsqueeze(1)
            case "linear":
                self.data = self.data.unsqueeze(1)
            case "maritime":
                datashape = self.data.shape
                reshaped = self.data.reshape(datashape[0], datashape[1]//2, 2) 
                self.data = reshaped.transpose(1,2)
            case "robot2" | "robot4" | "robot5":
                datashape = self.data.shape
                reshaped = self.data.reshape(datashape[0], datashape[1]//6, 6) 
                self.data = reshaped.transpose(1,2)[:, :3, :] #Choosing only the first 3 components of the dataset
            case "train":
                self.data = self.data.unsqueeze(1)
                # NOTE: all data must be in the format [samples, n_vars, n_traj_points]
        
        # Normalize the data if needed:
        if normalize:
            mean = self.data.mean(dim=(0, 2), keepdim=True) # computes the standard deviation across both dimension 0 (trajectories/samples) and dimension 2 (time points)
            std = self.data.std(dim=(0, 2), keepdim=True)
            # Avoid division by zero
            std = torch.clamp(std, min=1e-8)
            # Normalize
            self.data = (self.data - mean) / std

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleRNN(nn.Module):
    """
    Gated Recurrent Unit model for time series classification
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2, bidirectional=True):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * self.directions, 2)  # Binary classification
        
    def forward(self, x):

        # Reordering the input to feed into the gru
        x_reor = x.permute(0, 2, 1) # Now the shape is [samples, n_traj_points, n_vars]

        # GRU returns output, hidden
        _, hidden = self.gru(x_reor)
        
        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        # Need to reshape to separate layers and directions
        hidden = hidden.view(self.num_layers, self.directions, hidden.size(1), self.hidden_dim)

        # Get only the last layer's hidden states
        hidden = hidden[-1]  # shape: (num_directions, batch_size, hidden_dim)
        
        # Concatenate the forward and backward directions if bidirectional
        if self.bidirectional:
            forward, backward = hidden[0], hidden[1]  # Each shape: (batch_size, hidden_dim)
            hidden = torch.cat((forward, backward), dim=1)  # shape: (batch_size, 2*hidden_dim)
        else:
            hidden = hidden.squeeze(0)  # shape: (batch_size, hidden_dim)
        
        # Pass through fully connected layer
        out = self.fc(hidden)

        return out


def train_model(model, train_loader, val_loader=None, epochs=30, learning_rate=0.001, weight_decay=1e-5, device='cpu'):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of epochs to train
        learning_rate: Learning rate
        weight_decay: L2 regularization parameter
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model and training history
    """
    # Check if GPU is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track stats
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate epoch stats
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    
    return model, history


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Accuracy and predictions
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return accuracy, all_preds, all_labels


def load_and_prepare_data(dataset_name, batch_size=32, test_size=0.2):
    """
    Load and prepare data for a specific dataset
    
    Args:
        dataset_name: Name of the dataset ('linear', 'maritime', 'human', 'robot2', 'robot4', 'robot5', 'train')
        batch_size: Batch size for data loaders
        test_size: Proportion of the dataset to use for testing
        
    Returns:
        train_loader, test_loader, input_dim
    """
    # Load the data
    dataset = TimeSeriesDataset(dataset_name)
    
    # Get input dimension
    input_dim = dataset.data.shape[1] # Based on the shape: [samples, nvars, n_traj_points]
    
    # Handle special test_size cases
    if test_size == 0:
        # Use all data for training
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
        # Create an empty dataset for testing
        test_loader = DataLoader(
            dataset=torch.utils.data.Subset(dataset, []),
            batch_size=batch_size,
            shuffle=False
        )
    elif test_size == 1:
        # Use all data for testing
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False
        )
        # Create an empty dataset for training
        train_loader = DataLoader(
            dataset=torch.utils.data.Subset(dataset, []),
            batch_size=batch_size,
            shuffle=True
        )
    else:
        # Normal split
        train_indices, test_indices = train_test_split(
            range(len(dataset)), 
            test_size=test_size, 
            random_state=42,
            stratify=dataset.labels
        )
        
        train_loader = DataLoader(
            dataset=torch.utils.data.Subset(dataset, train_indices),
            batch_size=batch_size,
            shuffle=True
        )
        
        test_loader = DataLoader(
            dataset=torch.utils.data.Subset(dataset, test_indices),
            batch_size=batch_size,
            shuffle=False
        )
    
    return train_loader, test_loader, input_dim


if __name__ == "__main__":
    try:
        # Getting the name of the dataset to use
        #test_name = sys.argv[1]
        test_name = "robot4"
        # Training/testing phase
        phase = "full_train"
        #phase = sys.argv[2]

    except:
        raise RuntimeError("Wrong usage. The arguments must be: test_name, phase")

    if test_name not in {'linear', 'maritime', 'human', 'robot2', 'robot4', 'robot5', 'train'}:
        raise RuntimeError(f"{test_name} is not a correct test name")

    elif (phase == "test"):
        # Preparing the dataset
        train_loader, test_loader, input_dim = load_and_prepare_data(test_name, test_size=1)

        # Loading the model
        model = SimpleRNN(input_dim=input_dim)
        model.load_state_dict(torch.load(f'IR/data/data/{test_name}/model_state_dict.pth'))
        model.eval()

        # Evaluate
        accuracy, predictions, true_labels = evaluate_model(model, test_loader)

    elif (phase == "train_test"):
        # Preparing the dataset
        train_loader, test_loader, input_dim = load_and_prepare_data(test_name)

        # Create and train model
        model = SimpleRNN(input_dim=input_dim)
        trained_model, history = train_model(model, train_loader, test_loader, epochs=20)

        # Saving the model
        torch.save(model.state_dict(), f'IR/data/data/{test_name}/model_state_dict.pth')

        # Evaluate
        accuracy, predictions, true_labels = evaluate_model(trained_model, test_loader)
    
    elif (phase == "full_train"):
        # Preparing the dataset
        train_loader, test_loader, input_dim = load_and_prepare_data(test_name, test_size=0)

        # Create and train model
        model = SimpleRNN(input_dim=input_dim)
        trained_model, history = train_model(model, train_loader, test_loader, epochs=30)

        # Saving the model
        torch.save(model.state_dict(), f'IR/data/data/{test_name}/model_state_dict.pth')

    else:
        raise RuntimeError(f"{phase} is not a correct phase")