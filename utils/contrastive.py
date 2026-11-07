"""Class-based supervised contrastive regularisation for patient embeddings."""

import math

import torch
from torch.nn import functional as F


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Single-view SupCon over the current process's minibatch.

    Other patients with the same class are positives; self-pairs are excluded.
    Anchors without positives are skipped. A batch with no valid anchors
    contributes a differentiable zero. No cross-process or cross-step gathering
    is performed. Compute similarities in float32 even under mixed precision.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be finite and positive")
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("Expected features [B, D] and labels [B]")
    with torch.autocast(device_type=features.device.type, enabled=False):
        features = F.normalize(features.float(), dim=-1)
        n = features.shape[0]
        if n < 2:
            return features.sum() * 0.0
        self_pairs = torch.eye(n, device=features.device, dtype=torch.bool)
        positives = labels[:, None].eq(labels[None, :]) & ~self_pairs
        counts = positives.sum(dim=1)
        valid = counts > 0
        logits = (features @ features.T) / temperature
        denominator = logits.masked_fill(self_pairs, float('-inf')).logsumexp(dim=1)
        log_prob = logits - denominator[:, None]
        per_anchor = -log_prob.masked_fill(~positives, 0.0).sum(dim=1) / counts.clamp_min(1)
        return per_anchor.sum() / valid.sum().clamp_min(1)
