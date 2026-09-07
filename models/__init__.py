"""
Model architectures for post-calamity crop disease outbreak prediction.

The primary result is a classical RandomForest/XGBoost (training/classical_baselines.py),
not a model defined in this package — see results/fusion_model_results.json for why.
fusion_model_tabular.py holds the CNN-LSTM-style fusion model that was evaluated as an
alternative and reported as a documented negative finding.
"""
from models.fusion_model_tabular import TabularFusionModel, SpatialEncoder, TemporalEncoderWithAttention

__all__ = ["TabularFusionModel", "SpatialEncoder", "TemporalEncoderWithAttention"]
