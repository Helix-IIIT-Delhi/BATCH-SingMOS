"""Bhattacharyya Distance Computation"""

import torch
import torch.nn.functional as F


def bhattacharyya_distance(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute Bhattacharyya distance between two batches of features.

    B(P, Q) = -ln(∫√(P(x)Q(x))dx)

    For high-dimensional Gaussian approximation:
    B ≈ (1/8)(μ_p - μ_q)^T Σ^{-1}(μ_p - μ_q) + (1/2)ln(det(Σ)/√(det(Σ_p)det(Σ_q)))

    Args:
        p: Features from model P [batch_size, feature_dim]
        q: Features from model Q [batch_size, feature_dim]
        eps: Numerical epsilon for stability

    Returns:
        Bhattacharyya distances [batch_size]
    """
    batch_size = p.size(0)

    # Compute means
    mean_p = p.mean(dim=0)
    mean_q = q.mean(dim=0)

    # Compute covariances
    cov_p = torch.cov(p.T)
    cov_q = torch.cov(q.T)

    # Average covariance
    cov_avg = (cov_p + cov_q) / 2.0

    # Mean difference term
    mean_diff = mean_p - mean_q

    try:
        # Compute Mahalanobis distance
        cov_avg_inv = torch.inverse(cov_avg + eps * torch.eye(cov_avg.size(0), device=cov_avg.device))
        mahal_term = torch.matmul(mean_diff, torch.matmul(cov_avg_inv, mean_diff))
    except:
        # Fallback to simpler distance
        mahal_term = torch.norm(mean_diff, p=2) ** 2

    # Determinant term
    try:
        det_p = torch.det(cov_p + eps * torch.eye(cov_p.size(0), device=cov_p.device))
        det_q = torch.det(cov_q + eps * torch.eye(cov_q.size(0), device=cov_q.device))
        det_avg = torch.det(cov_avg + eps * torch.eye(cov_avg.size(0), device=cov_avg.device))

        # Avoid log of negative numbers
        det_p = torch.clamp(det_p, min=eps)
        det_q = torch.clamp(det_q, min=eps)
        det_avg = torch.clamp(det_avg, min=eps)

        det_term = 0.5 * torch.log(det_avg / torch.sqrt(det_p * det_q))
    except:
        det_term = 0.0

    # Bhattacharyya distance
    distance = 0.125 * mahal_term + det_term

    return torch.clamp(distance, min=0)


def bhattacharyya_coefficient(
    p: torch.Tensor,
    q: torch.Tensor
) -> torch.Tensor:
    """
    Compute Bhattacharyya coefficient (similarity measure).

    BC = ∫√(P(x)Q(x))dx

    Args:
        p: Probability distribution or features
        q: Probability distribution or features

    Returns:
        Bhattacharyya coefficient [0, 1]
    """
    # Normalize to probability distributions
    p_norm = F.softmax(p, dim=-1)
    q_norm = F.softmax(q, dim=-1)

    # Compute coefficient
    bc = torch.sum(torch.sqrt(p_norm * q_norm), dim=-1)

    return bc
