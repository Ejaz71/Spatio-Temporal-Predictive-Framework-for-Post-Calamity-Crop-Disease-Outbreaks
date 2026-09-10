"""
CNN-LSTM-style fusion model adapted to real data.

Spatial branch: proposal Section 6.1 describes a convolutional branch over remote-
sensing raster patches. Under Option A (the primary modeling unit per Section 6.2 —
one feature vector per district-season event, not raster patches), there is no 2D
pixel grid for a literal CNN to convolve over: this is a small MLP over the 5 real
remote-sensing summary features (SAR VV/VH, NDVI, NDWI, LST) for that event.

Temporal branch: originally this ran over 9 pre-aggregated meteorological summary
scalars reshaped into a fake length-9 "sequence" (proposal Section 8's suggested
fallback). It now runs over the REAL 90-day daily NASA POWER sequence for that
event's window (data/extract_daily_sequences.py, recovered from already-cached API
responses, not re-fetched) — an LSTM operating on an actual daily series rather than
a pseudo-sequence of summary statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SPATIAL_FEATURES = ["sar_vv_db_mean", "sar_vh_db_mean", "ndvi_mean", "ndwi_mean", "lst_celsius_mean",
                    "water_extent_frac"]
TEMPORAL_FEATURES = [
    "precip_mean_mm", "precip_max_mm", "precip_sum_mm", "precip_anomaly_mm",
    "rh_mean_pct", "rh_max_pct", "temp_mean_c", "vpd_mean_kpa", "wet_persistence_max_days",
    "monsoon_precip_sum_mm", "monsoon_precip_anomaly_mm", "monsoon_rh_mean_pct",
    "monsoon_temp_mean_c", "monsoon_vpd_mean_kpa", "monsoon_wet_persistence_max_days",
]
DAILY_SEQUENCE_FEATURES = ["precip_mm", "precip_anomaly_mm", "temp_c", "rh_pct", "vpd_kpa", "wet_persistence_days"]


class SpatialEncoder(nn.Module):
    def __init__(self, in_features=5, hidden_dim=12, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class TemporalEncoderWithAttention(nn.Module):
    def __init__(self, n_daily_features=6, hidden_dim=12, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Linear(n_daily_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Parameter(torch.randn(hidden_dim) * 0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, seq_len=90, n_daily_features) — the real daily meteorological series.
        proj = F.gelu(self.input_proj(x))
        lstm_out, _ = self.lstm(proj)  # (B, seq_len, hidden)
        u = torch.tanh(self.attn_query(lstm_out))  # (B, seq_len, hidden)
        scores = torch.matmul(u, self.attn_v)  # (B, seq_len)
        attn_weights = F.softmax(scores, dim=-1)
        pooled = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (B, hidden)
        return self.dropout(pooled), attn_weights


class TabularFusionModel(nn.Module):
    def __init__(self, spatial_dim=5, n_daily_features=6, hidden_dim=12, dropout=0.4):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(spatial_dim, hidden_dim, dropout)
        self.temporal_encoder = TemporalEncoderWithAttention(n_daily_features, hidden_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, spatial, temporal):
        z_spatial = self.spatial_encoder(spatial)
        z_temporal, attn_weights = self.temporal_encoder(temporal)
        fused = torch.cat([z_spatial, z_temporal], dim=-1)
        logit = self.classifier(fused).squeeze(-1)
        return logit, attn_weights
