import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.nn.utils import parametrizations as param
from torch.utils.data import random_split
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os
import Dash 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadData(file_paths: list, timesteps: int) -> tuple:
    df_list = []
    timestamp_list = []  # List to store timestamps from each CSV file being read 
    
    for path in file_paths:
        df = pd.read_csv(path, skiprows=3)

        # Extract relevant columns
        machine_id_col = 'Link'
        axis_col = '_field'
        value_col = '_value'
        time_col = '_time'

        # Filter relevant columns
        filtered_df = df[[machine_id_col, axis_col, value_col, time_col]]

        # Pivot data to have separate columns for each Axis and its corresponding values
        pivoted_df = filtered_df.pivot_table(index=[machine_id_col, time_col], columns=[axis_col], values=value_col)

        # Reset the index to make Machine ID and Time as regular columns
        pivoted_df.reset_index(inplace=True)

        # Convert time column to datetime type
        pivoted_df[time_col] = pd.to_datetime(pivoted_df[time_col])

        # Sort DataFrame by Machine ID and Time
        pivoted_df.sort_values(by=[machine_id_col, time_col], inplace=True)

        # Fill missing values using forward fill
        pivoted_df.ffill(inplace=True)  
        pivoted_df.dropna(inplace=True)
        
        # Ensure we only keep timestamps that have corresponding data after dropping NaN
        timestamps = pivoted_df[time_col]

        # Select first four features for processing
        first_four_features = [f'Axis Z value{i}' for i in range(4)]
        feature_df = pivoted_df[first_four_features]

        # Append the feature DataFrame to the list
        df_list.append(feature_df)

        # Append timestamps to the timestamp list
        timestamp_list.append(timestamps)

    # Combine time-series from all files
    concatenated_time_series = np.concatenate([df.values for df in df_list])

    # Combine timestamps from all files
    concatenated_timestamps = pd.concat(timestamp_list).reset_index(drop=True).values


 
    # Normalize the concatenated time series data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(concatenated_time_series)

    normalized_tensor = torch.FloatTensor(normalized_data)

    

    return normalized_tensor, normalized_data, concatenated_timestamps



path = 'Path To Your Trainingdata'

csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

test_csv_file = 'Used for inference'

test_csv_files = [test_csv_file]

# List all CSV files in the directory
csv_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

# Remove the specific test CSV file from the list of CSV files
train_val_csv_files = [file for file in csv_files if file != test_csv_file]

normalized_tensor, normalized_data, timestamps = loadData(train_val_csv_files, 0)

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i+sequence_length]
        sequences.append(sequence)
    return torch.stack(sequences)  


sequence_length = 100
sequences = create_sequences(normalized_tensor, sequence_length)
sequences = torch.transpose(sequences,1,2)

def train_val_split(sequences, train_size, val_size):
    assert train_size + val_size <= 1, "Sum of train and validation sizes should be 1 or less."

    total_sequences = sequences.shape[0]
    train_end = int(train_size * total_sequences)
    val_end = train_end + int(val_size * total_sequences)

    train_dataset = sequences[:train_end]
    val_dataset = sequences[train_end:val_end]

    return train_dataset, val_dataset


# Define the sizes of the splits
train_size = 0.7 
val_size = 0.2    


# Split the data
train_dataset, val_dataset = train_val_split(sequences, train_size, val_size)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
validation_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=True)

# Load and prepare test data
normalized_test_tensor, normalized_test_data, test_timestamps = loadData(test_csv_files, 0)
test_sequences = create_sequences(normalized_test_tensor, sequence_length)
test_sequences = torch.transpose(test_sequences,1,2)
test_loader = DataLoader(test_sequences, shuffle=False, batch_size=batch_size, drop_last=True)



num_features_to_plot = 4 # Modify for your parametercount
features_to_plot = normalized_data[:, :num_features_to_plot]


time_steps = np.arange(features_to_plot.shape[0])

#Architecture begins
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        
        x = self.conv(x)
        x = x[:, :, :x.size(2) - self.padding]
        
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        # First convolution layer
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.conv1.conv = param.weight_norm(self.conv1.conv)
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution layer
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.conv2.conv = param.weight_norm(self.conv2.conv)
        self.dropout2 = nn.Dropout(dropout)

        # Downsample layer for residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()


    def init_weights(self):
        self.conv1.conv.weight.data.normal_(0, 0.01)
        self.conv2.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        
        
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        

        out = self.relu(self.conv2(out))
        out = self.dropout2(out)
        

        out = self.relu(out + res)
        return out
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        
        for i, layer in enumerate(self.network):
            x = layer(x)
            
        return x

class SelfAttention(nn.Module): 
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
    

        # Reshape for multi-head attention
        values = values.reshape(N, -1, self.heads, self.head_dim) 
        keys = keys.reshape(N, -1, self.heads, self.head_dim) 
        queries = query.reshape(N, -1, self.heads, self.head_dim)

        # Apply linear transformations
        values = self.values(values)
        keys = self.keys(keys)

        queries = self.queries(queries)

        # Attention mechanism
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))
        attention_weights = torch.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention_weights, values]).reshape(
            N, -1, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out, attention_weights
    
class TCNWithAttention(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, embed_size, heads):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout) #The first TCN-Attention handshake
        self.attention = SelfAttention(embed_size, heads)

        self.fc = nn.Linear(embed_size, num_inputs)

    def forward(self, x, mask):
        tcn_out = self.tcn(x)
        tcn_out_flat = tcn_out.view(tcn_out.size(0), -1)
        attn_out, attn_weights = self.attention(tcn_out_flat, tcn_out_flat, tcn_out_flat, mask)
        attn_out = attn_out.view(tcn_out.size(0), tcn_out.size(2), -1)
        out = self.fc(attn_out)
        return out, attn_weights


num_inputs = 4  
num_channels = [1, 64, 128]  
model = TCNWithAttention(num_inputs=num_inputs, num_channels=num_channels, kernel_size=3, dropout=0.2, embed_size=128, heads=4)
model.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs):
    print(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for batch_idx, sequences in enumerate(train_loader):
            sequences = sequences.to(device)  
            optimizer.zero_grad()
            reconstructed, attn_weights = model(sequences, mask=None)
            reconstructed = torch.transpose(reconstructed, 1,2)
            #print('JGHGHJGJGJ',sequences.shape)
            #print('HKHKHKHKHK',reconstructed.shape)
            
            loss = criterion(reconstructed, sequences.squeeze(1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Training Batch Loss: {loss.item()}')

        average_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Training Loss: {average_train_loss}')

        # Validation phase
        validate_model(model, validation_loader, criterion)

def validate_model(model, validation_loader, criterion):
    print("Starting validation")
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for sequences in validation_loader:
            sequences = sequences.to(device) 
            reconstructed, attn_weights = model(sequences, mask=None)
            reconstructed = torch.transpose(reconstructed, 1,2)
            loss = criterion(reconstructed, sequences.squeeze(1))
            total_val_loss += loss.item()
            print(f'Validation Batch Loss: {loss.item()}')

    average_val_loss = total_val_loss / len(validation_loader)
    print(f'Validation Loss: {average_val_loss}\n')

def test_model(model, test_loader, criterion):
    print("Starting testing")
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for sequences in test_loader:
            sequences = sequences.to(device)
            reconstructed, attn_weights = model(sequences, mask=None)
            reconstructed = torch.transpose(reconstructed, 1,2)
            loss = criterion(reconstructed, sequences.squeeze(1))
            total_test_loss += loss.item()
            print(f'Test Batch Loss: {loss.item()}')

    average_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {average_test_loss}\n')

def calculate_reconstruction_errors(model, data_loader):
    print("Calculating reconstruction errors")
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for sequences in data_loader:
            sequences = sequences.to(device)
            reconstructed, attn_weights = model(sequences, mask=None)
            reconstructed = torch.transpose(reconstructed, 1,2)
            error = ((sequences - reconstructed) ** 2).mean(dim=(1, 2))
            reconstruction_errors.extend(error.cpu().numpy())
    return np.array(reconstruction_errors)

def detect_anomalies(reconstruction_errors, threshold):
    print("Detecting anomalies")
    high_error_indices = np.where(reconstruction_errors > threshold)[0]
    return high_error_indices


# Train the model
train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=5)


# After training and validation are complete, evaluate on the test set
test_model(model, test_loader, criterion)

# Downstream task: After testing, we calculate reconstruction errors and detect anomalies on the test set
reconstruction_errors = calculate_reconstruction_errors(model, test_loader)  # Use test_loader here
threshold = np.percentile(reconstruction_errors, 95)
print(f"Anomaly detection threshold (99th percentile): {threshold}")

anomaly_indices = detect_anomalies(reconstruction_errors, threshold)  # Detect anomalies in the test set
print(f"Anomalies detected at indices: {anomaly_indices}")


sequences = next(iter(test_loader)).to(device)
with torch.no_grad():
    _, attn_weights = model(sequences, mask=None)

# Visualize the attention weights for the first example and first head
attention = attn_weights[0, 0].cpu().numpy()
sns.heatmap(attention, cmap='viridis')
plt.title('Attention weights heatmap')
plt.xlabel('Key/query positions')
plt.ylabel('Output positions')
plt.show()


features_to_plot = normalized_data[:, :4]
num_features_to_plot = 4 

# Convert indices to boolean mask for easy anomaly plotting
anomaly_mask = np.zeros(len(normalized_data), dtype=bool)
anomaly_mask[anomaly_indices] = True


temperature_df = pd.read_csv('C:\\Users\\Stell\\Desktop\\Navigation\\.venv\\Code Leine Linde\\bearing temp\\bearing temp rig 2 run 4 non-envelope 4 features.csv', skiprows=3)
temperature_df['_time'] = pd.to_datetime(temperature_df['_time'])


test_timestamps = pd.to_datetime(test_timestamps)


axes = ['Axis X', 'Axis Y', 'Axis Z']
values_per_axis = 4 # Assuming there are 4 values per axis
feature_names = [f'{axis} value{i}' for axis in axes for i in range(values_per_axis)]



anomaly_mask_test = np.zeros(len(test_timestamps), dtype=bool)
anomaly_mask_test[anomaly_indices] = True



subplot_height = 0.3

specs = [[{"secondary_y": True} for _ in range(4)] for _ in range(3)]
fig = make_subplots(
    rows=3, 
    cols=4, 
    subplot_titles=feature_names, 
    specs=specs,
    horizontal_spacing=0.01,  # Adjust horizontal spacing here
    vertical_spacing=0.04  # And adjust vertical spacing if needed
)

for i, feature_name in enumerate(feature_names):
    row = (i // 4) +1
    col = (i % 4) +1
    
    
    fig.add_trace(
        go.Scatter(
            x=test_timestamps, 
            y=normal_train[:, i],
            mode='lines',
            name=feature_name,
            line=dict(width=2),
        ),
        row=row,
        col=col,
        secondary_y=False
    )
    
    
    fig.add_trace(
        go.Scatter(
            x=test_timestamps[anomaly_mask_test],
            y=normal_train[anomaly_mask_test, i], #Replace with the trainingdata you wish to plot for 
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=5),
            showlegend=i==0  
        ),
        row=row,
        col=col,
        secondary_y=False
    )
    
    
    fig.add_trace(
        go.Scatter(
            x=temperature_df['_time'],
            y=temperature_df['_value'],
            mode='lines',
            name='Temperature',
            line=dict(color='orange', dash='dot', width=2),
            showlegend=i==0  #
        ),
        row=row,
        col=col,
        secondary_y=True
    )
    
    
    fig.update_yaxes(
        title_text='',
        showticklabels=False,
        row=row,
        col=col,
        secondary_y=True
    )



fig.update_layout(
    height=1600,
    title_text='Time Series Data with Anomalies and Temperature Overlay For Isolation Forest',
    showlegend=True,
    legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="right", 
        x=1
    ),
    margin=dict(l=40, r=40, t=100, b=40)  
)

fig.show()
