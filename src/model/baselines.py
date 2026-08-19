import torch
import torch.nn as nn
import math

class StandardLSTM(nn.Module):
    """
    Standard Baseline LSTM for GEO Electron Flux Forecasting.
    Mimics traditional data-driven models (e.g., Chu et al., 2021) without 
    adaptive delays or physics penalties. Assumes OMNI static 1-hour delay.
    """
    def __init__(self, n_sw_features, seq_len=72, hidden_dim=64, num_layers=2, n_horizons=3):
        super().__init__()
        # Input features: SW + historical flux (1)
        self.lstm = nn.LSTM(
            input_size=n_sw_features + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, n_horizons)

    def forward(self, x_sw, x_flux, **kwargs):
        # Sanitize inputs to prevent NaNs from propagating through LSTM
        x_sw = torch.nan_to_num(x_sw, nan=0.0)
        x_flux = torch.nan_to_num(x_flux, nan=0.0)
        # x_sw: [B, S, F], x_flux: [B, S, 1]
        x = torch.cat([x_sw, x_flux], dim=-1)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = out[:, -1, :]
        preds = self.fc(last_out)
        return {
            "flux_pred": preds,
            "delay_loss": torch.tensor(0.0, device=preds.device, requires_grad=True), # Dummy loss
            "log_var": torch.zeros_like(preds), # No uncertainty
        }

class StandardMLP(nn.Module):
    """
    Standard MLP / Deep Neural Network (e.g., Wei et al., 2018).
    Flattens the temporal sequence into a single vector.
    """
    def __init__(self, n_sw_features, seq_len=72, hidden_dim=128, n_horizons=3):
        super().__init__()
        input_dim = (n_sw_features + 1) * seq_len
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_horizons)
        )

    def forward(self, x_sw, x_flux, **kwargs):
        x_sw = torch.nan_to_num(x_sw, nan=0.0)
        x_flux = torch.nan_to_num(x_flux, nan=0.0)
        # Flatten: [B, S, F] -> [B, S*F]
        x = torch.cat([x_sw, x_flux], dim=-1)
        x = x.view(x.size(0), -1)
        preds = self.net(x)
        return {
            "flux_pred": preds,
            "delay_loss": torch.tensor(0.0, device=preds.device, requires_grad=True),
            "log_var": torch.zeros_like(preds),
        }

class StandardCNN(nn.Module):
    """
    1D CNN / Autoregressive approximation (e.g., Shin et al., 2020).
    Uses 1D convolutions over the sequence length.
    """
    def __init__(self, n_sw_features, seq_len=72, channels=64, n_horizons=3):
        super().__init__()
        in_channels = n_sw_features + 1
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Pools to [B, C*2, 1]
        )
        self.fc = nn.Linear(channels * 2, n_horizons)

    def forward(self, x_sw, x_flux, **kwargs):
        x_sw = torch.nan_to_num(x_sw, nan=0.0)
        x_flux = torch.nan_to_num(x_flux, nan=0.0)
        x = torch.cat([x_sw, x_flux], dim=-1)
        # Conv1d expects [B, C, S]
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = out.squeeze(-1)
        preds = self.fc(out)
        return {
            "flux_pred": preds,
            "delay_loss": torch.tensor(0.0, device=preds.device, requires_grad=True),
            "log_var": torch.zeros_like(preds),
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class VanillaTransformer(nn.Module):
    """
    Standard Transformer Network (e.g., Pinto et al., 2022).
    Uses standard temporal attention, no spatial variable attention, 
    no adaptive delay.
    """
    def __init__(self, n_sw_features, seq_len=72, d_model=64, nhead=4, num_layers=3, n_horizons=3):
        super().__init__()
        self.embedding = nn.Linear(n_sw_features + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, n_horizons)

    def forward(self, x_sw, x_flux, **kwargs):
        # Sanitize inputs to prevent NaNs from propagating through Transformer
        x_sw = torch.nan_to_num(x_sw, nan=0.0)
        x_flux = torch.nan_to_num(x_flux, nan=0.0)
        
        x = torch.cat([x_sw, x_flux], dim=-1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        # Use the representation of the last token
        last_out = out[:, -1, :]
        preds = self.fc(last_out)
        return {
            "flux_pred": preds,
            "delay_loss": torch.tensor(0.0, device=preds.device, requires_grad=True), # Dummy loss
            "log_var": torch.zeros_like(preds), # No uncertainty
        }
