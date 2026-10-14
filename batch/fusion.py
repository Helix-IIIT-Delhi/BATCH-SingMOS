"""BATCH Fusion Framework for SingMOS Prediction"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from .distance import bhattacharyya_distance


class BATCHFusion(nn.Module):
    """
    BATCH: Bhattacharyya Distance-based Fusion Framework

    Fuses multiple pre-trained model outputs using Bhattacharyya distance
    for singing voice quality assessment (SingMOS prediction).

    Args:
        models: List of pre-trained model extractors
        fusion_dim: Dimension of fused representation
        learnable_weights: Whether to learn fusion weights
    """

    def __init__(
        self,
        models: List[nn.Module],
        fusion_dim: int = 512,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.fusion_dim = fusion_dim
        self.learnable_weights = learnable_weights

        # Learnable fusion weights
        if learnable_weights:
            self.fusion_weights = nn.Parameter(
                torch.ones(self.num_models) / self.num_models
            )
        else:
            self.register_buffer(
                "fusion_weights",
                torch.ones(self.num_models) / self.num_models
            )

        # Projection layers for each model
        self.projections = nn.ModuleList([
            nn.Linear(model.output_dim if hasattr(model, 'output_dim') else 512, fusion_dim)
            for model in models
        ])

        # SingMOS prediction head
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output in [0, 1], scale to [0, 5]
        )

    def forward(self, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with BATCH fusion.

        Args:
            audio_features: Input audio features

        Returns:
            Dict with:
                - singmos_score: Predicted SingMOS (0-5)
                - fused_features: Fused representation
                - model_outputs: Individual model outputs
        """
        # Extract features from each model
        model_outputs = []
        projected_features = []

        for i, (model, proj) in enumerate(zip(self.models, self.projections)):
            features = model(audio_features)
            model_outputs.append(features)
            proj_features = proj(features)
            projected_features.append(proj_features)

        # Compute Bhattacharyya distances between model outputs
        distances = self._compute_bhattacharyya_distances(projected_features)

        # Compute fusion weights based on distances
        fusion_weights = self._compute_fusion_weights(distances)

        # Fuse representations
        fused_features = torch.zeros_like(projected_features[0])
        for i, feat in enumerate(projected_features):
            fused_features += fusion_weights[:, i:i+1] * feat

        # Predict SingMOS
        singmos_logit = self.predictor(fused_features)
        singmos_score = singmos_logit * 5.0  # Scale to [0, 5]

        return {
            "singmos_score": singmos_score.squeeze(-1),
            "fused_features": fused_features,
            "model_outputs": model_outputs,
            "fusion_weights": fusion_weights,
        }

    def _compute_bhattacharyya_distances(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute Bhattacharyya distances between feature pairs"""
        distances = torch.zeros(
            features[0].size(0),
            self.num_models,
            self.num_models,
            device=features[0].device
        )

        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                dist = bhattacharyya_distance(features[i], features[j])
                distances[:, i, j] = dist
                distances[:, j, i] = dist

        return distances

    def _compute_fusion_weights(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute fusion weights from Bhattacharyya distances"""
        batch_size = distances.size(0)

        # Initialize with learned weights or uniform
        weights = self.fusion_weights.unsqueeze(0).expand(batch_size, -1)

        # Adjust weights based on distances (complementarity)
        for i in range(self.num_models):
            avg_distance = distances[:, i, :].mean(dim=1, keepdim=True)
            weights[:, i] = weights[:, i] * (1.0 + avg_distance)

        # Normalize weights
        weights = torch.softmax(weights, dim=1)

        return weights
