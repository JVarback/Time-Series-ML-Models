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
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio.plotly as py
import chart_studio

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
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
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
    

        
        values = values.reshape(N, -1, self.heads, self.head_dim) 
        keys = keys.reshape(N, -1, self.heads, self.head_dim) 
        queries = query.reshape(N, -1, self.heads, self.head_dim)

        

        
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

    def forward(self, x, mask = None, return_latent=False):
        tcn_out = self.tcn(x)
        tcn_out_flat = tcn_out.view(tcn_out.size(0), -1)
        attn_out, attn_weights = self.attention(tcn_out_flat, tcn_out_flat, tcn_out_flat, mask)
        attn_out = attn_out.view(tcn_out.size(0), tcn_out.size(2), -1)
        out = self.fc(attn_out)

        return out, attn_weights

# Initialize the model
num_inputs = 12
num_channels = [12, 64, 128]  
kernel_size = 3
dropout = 0.2827
embed_size = 128
heads = 4


sequence_length = 200
step_size = 1
batch_size = 128



def init_model():
    print(f'init_model start')
    model = TCNWithAttention(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout, embed_size=embed_size, heads=heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.6835084022669533e-05)



    # Load the trained model weights
    checkpoint_path = 'TCNATT_checkpoint_epoch_15.pth'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval() 

    return model



def loadData(file_paths: list, timesteps: int, four_features) -> tuple:
    df_list = []
    timestamp_list = []  
    
    for path in file_paths:
        df = pd.read_csv(path, skiprows=3)

        # Extract relevant columns
        machine_id_col = 'Link'
        axis_col = '_field'
        value_col = '_value'
        time_col = '_time'

        

        
        filtered_df = df[[machine_id_col, axis_col, value_col, time_col]]



        
        pivoted_df = filtered_df.pivot_table(index=[machine_id_col, time_col], columns=[axis_col], values=value_col)

        
        pivoted_df.reset_index(inplace=True)

        
        pivoted_df[time_col] = pd.to_datetime(pivoted_df[time_col])

        
        pivoted_df.sort_values(by=[machine_id_col, time_col], inplace=True)

        
        pivoted_df.ffill(inplace=True)  
        pivoted_df.dropna(inplace=True)  
        
        
        timestamps = pivoted_df[time_col]

        
        if four_features:
        
            first_four_features = [f'Axis Z value{i}' for i in range(4)]
            feature_df = pivoted_df[first_four_features]
        else:
            features = ['Axis X', 'Axis Y', 'Axis Z']
            twelve_features = [f'{axis} value{i}' for axis in features for i in range(4)]
            feature_df = pivoted_df[twelve_features]


        
        df_list.append(feature_df)

        
        timestamp_list.append(timestamps)


    
    concatenated_time_series = np.concatenate([df.values for df in df_list])

    
    concatenated_timestamps = pd.concat(timestamp_list).reset_index(drop=True).values

 
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(concatenated_time_series)
    
    # Final sanity check: Ensure the number of timestamps matches the number of data points
    if len(concatenated_timestamps) != len(normalized_data):
        raise ValueError("The length of timestamps does not match the length of the normalized data.")

    normalized_tensor = torch.FloatTensor(normalized_data)

    

    return normalized_tensor, normalized_data, concatenated_timestamps



def create_sequences(data, sequence_length, step_size):
    
    sequences = []
    for i in range(0, len(data) - sequence_length+(sequence_length)+1, int(step_size)):
        sequence = data[i:i+sequence_length]
        if len(sequence) < sequence_length: 
            break

        sequences.append(sequence)
        
        
    return torch.stack(sequences)



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



def start(path, model):
    test_csv_file = [path]
 
    
    normalized_test_tensor, normalized_test_data, test_timestamps = loadData(test_csv_file, 0, False)
    test_sequences = create_sequences(normalized_test_tensor, sequence_length, step_size)
    test_sequences = torch.transpose(test_sequences,1,2)
    test_loader = DataLoader(test_sequences, shuffle=False, batch_size=batch_size, drop_last=True)

    reconstruction_errors = calculate_reconstruction_errors(model, test_loader) 
    threshold = np.percentile(reconstruction_errors, 95)
    print(f"Anomaly detection threshold (95th percentile): {threshold}")

    anomaly_indices = detect_anomalies(reconstruction_errors, threshold) 
    print(f"Anomalies detected at indices: {anomaly_indices}")


    test_timestamps = pd.to_datetime(test_timestamps)


    axes = ['Axis X', 'Axis Y', 'Axis Z']
    values_per_axis = 4  
    feature_names = [f'{axis} value{i}' for axis in axes for i in range(values_per_axis)]


    anomaly_mask_test = np.zeros(len(test_timestamps), dtype=bool)
    anomaly_mask_test[anomaly_indices] = True

    return anomaly_indices