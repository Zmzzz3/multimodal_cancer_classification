"""Multimodal and unimodal ABMIL/MLP tumour classifiers."""

import torch
from torch import nn
from timm.models import register_model


def _wsi_projection(input_dim, width):
    """Project each patch independently, preserving the patch dimension."""
    layers = []
    for in_features in (input_dim, width):
        layers.extend([
            nn.Linear(in_features, width), nn.LayerNorm(width),
            nn.GELU(), nn.Dropout(0.1),
        ])
    return nn.Sequential(*layers)


class AttentionPool(nn.Module):
    """Project patch features and pool each bag with gated attention.

    Args:
        input_dim: Number of features per WSI patch.
        width: Width of the projected patches and pooled embedding.
    """

    def __init__(self, input_dim, width):
        super().__init__()
        self.project = _wsi_projection(input_dim, width)
        self.tanh_gate = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Dropout(0.25))
        self.sigmoid_gate = nn.Sequential(nn.Linear(width, width), nn.Sigmoid(), nn.Dropout(0.25))
        self.score = nn.Linear(width, 1)

    def forward(self, bag):
        """Return a pooled embedding for each WSI bag.

        Args:
            bag: Floating tensor of shape [B, N, input_dim], with N patches.

        Returns:
            A floating tensor of shape [B, width].
        """
        patches = self.project(bag)
        gates = self.tanh_gate(patches) * self.sigmoid_gate(patches)
        # Normalize over patches so each bag's attention weights sum to one.
        weights = self.score(gates).softmax(dim=1)
        return torch.bmm(weights.transpose(1, 2), patches).squeeze(1)


def _rna_encoder(input_dim, width):
    """Build the RNA MLP shared by the paired and RNA-only baselines."""
    layers = []
    for in_features in (input_dim, width):
        layers.extend([
            nn.Linear(in_features, width), nn.LayerNorm(width),
            nn.ReLU(), nn.Dropout(0.2),
        ])
    layers.append(nn.Linear(width, width))
    return nn.Sequential(*layers)


class BaselineClassifier(nn.Module):
    """Fuse WSI attention pooling and an RNA MLP by concatenation.

    Args:
        rna_embed_dim: Number of supplied RNA features per patient.
        wsi_embed_dim: Number of features per WSI patch.
        embed_dim: Output width of each modality encoder.
        num_classes: Number of output class scores.
        fusion: Fusion rule; only "concat" is supported.

    Raises:
        ValueError: If fusion is not "concat".
    """

    def __init__(self, rna_embed_dim, wsi_embed_dim=768, embed_dim=768, num_classes=2, fusion="concat"):
        super().__init__()
        if fusion != "concat":
            raise ValueError("This baseline uses concatenation fusion")
        self.num_classes = num_classes
        self.wsi_encoder = AttentionPool(wsi_embed_dim, embed_dim)
        self.rna_encoder = _rna_encoder(rna_embed_dim, embed_dim)
        self.head = nn.Linear(2 * embed_dim, num_classes)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        """Return class logits from paired RNA and WSI features.

        Args:
            x: Two floating tensors in [RNA, WSI] order, with shapes
                [B, rna_embed_dim] and [B, N, wsi_embed_dim].

        Returns:
            Unnormalized logits of shape [B, num_classes]. The supplied
            configuration orders the two classes as LUAD, LUSC.
        """
        rna, wsi = x
        embedding = torch.cat([self.wsi_encoder(wsi), self.rna_encoder(rna)], dim=-1)
        return self.head(embedding)


class GatedMultimodalUnit(nn.Module):
    """Patient- and feature-specific mixing of WSI and RNA representations.

    Gate values near one favour WSI; values near zero favour RNA.
    Both inputs inform the sigmoid gate, while separate tanh projections
    place the two modalities into the shared output space.
    """

    def __init__(self, width):
        super().__init__()
        self.wsi_transform = nn.Linear(width, width)
        self.rna_transform = nn.Linear(width, width)
        self.gate = nn.Linear(2 * width, width)

    def forward(self, wsi, rna):
        gate = torch.sigmoid(self.gate(torch.cat([wsi, rna], dim=-1)))
        return (gate * torch.tanh(self.wsi_transform(wsi))
                + (1.0 - gate) * torch.tanh(self.rna_transform(rna)))


class RNACrossAttentionClassifier(nn.Module):
    """Use one RNA query to pool WSI patches through cross-attention.

    Reuses the baseline RNA MLP and patch projection. Cross-attention replaces
    gated ABMIL pooling: RNA supplies Q, and projected WSI patches supply K/V.
    The attended WSI vector and RNA embedding are fused by concatenation or
    GMU before the linear classifier and optional contrastive projector.
    Inputs follow the trainer's [RNA, WSI] convention.

    num_heads defaults to one; embed_dim must be divisible by num_heads.
    No positional encoding is used, so patch order does not affect evaluation.
    Bags contain real or resampled patches, not zero-padding (as in the loader).
    """

    def __init__(self, rna_embed_dim, wsi_embed_dim=768, embed_dim=768,
                 num_classes=2, fusion="concat", num_heads=1,
                 attention_dropout=0.0, contrastive_dim=0):
        super().__init__()
        if fusion not in ("concat", "gmu"):
            raise ValueError("RNA cross-attention fusion must be 'concat' or 'gmu'")
        if num_heads < 1 or embed_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide embed_dim")
        self.num_classes = num_classes
        self.wsi_project = _wsi_projection(wsi_embed_dim, embed_dim)
        self.rna_encoder = _rna_encoder(rna_embed_dim, embed_dim)
        # Includes learned Q, K, V projections and an output projection.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attention_dropout, batch_first=True,
        )
        self.gmu = GatedMultimodalUnit(embed_dim) if fusion == "gmu" else None
        fused_dim = embed_dim if self.gmu is not None else 2 * embed_dim
        self.head = nn.Linear(fused_dim, num_classes)

        if not isinstance(contrastive_dim, int) or contrastive_dim < 0:
            raise ValueError("contrastive_dim must be a non-negative integer")
        self.contrastive_projector = (
            nn.Sequential(
                nn.Linear(fused_dim, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, contrastive_dim),
            ) if contrastive_dim else None
        )

    def get_classifier(self):
        return self.head

    def forward(self, x, return_contrastive=False):
        """Return logits, optionally paired with training projection features.

        Default inference skips the projection head. L2 normalisation is done
        in float32 by the contrastive loss for mixed-precision stability.
        """
        rna, wsi = x
        rna_embedding = self.rna_encoder(rna)
        patches = self.wsi_project(wsi)
        attended_wsi, _ = self.cross_attention(
            query=rna_embedding.unsqueeze(1),
            key=patches,
            value=patches,
            need_weights=False,
        )
        wsi_embedding = attended_wsi.squeeze(1)
        embedding = (self.gmu(wsi_embedding, rna_embedding) if self.gmu is not None
                     else torch.cat([wsi_embedding, rna_embedding], dim=-1))
        logits = self.head(embedding)
        if return_contrastive:
            if self.contrastive_projector is None:
                raise ValueError("Set contrastive_dim > 0 to request contrastive features")
            return logits, self.contrastive_projector(embedding)
        return logits


class WSIOnlyClassifier(nn.Module):
    """Gated ABMIL followed by a linear classifier, using only WSI features.

    Accepts either [RNA, WSI] or a WSI tensor [B, N, wsi_embed_dim].
    rna_embed_dim and fusion are accepted for baseline-config compatibility;
    no RNA encoder or multimodal fusion is instantiated.
    """

    def __init__(self, rna_embed_dim=None, wsi_embed_dim=768, embed_dim=768,
                 num_classes=2, fusion="concat"):
        super().__init__()
        self.num_classes = num_classes
        self.wsi_encoder = AttentionPool(wsi_embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        wsi = x[1] if isinstance(x, (list, tuple)) else x
        return self.head(self.wsi_encoder(wsi))


class RNAOnlyClassifier(nn.Module):
    """RNA MLP followed by a linear classifier, using only RNA features.

    Accepts either [RNA, WSI] or an RNA tensor [B, rna_embed_dim].
    wsi_embed_dim and fusion are accepted for baseline-config compatibility;
    no WSI encoder or multimodal fusion is instantiated.
    """

    def __init__(self, rna_embed_dim, wsi_embed_dim=768, embed_dim=768,
                 num_classes=2, fusion="concat"):
        super().__init__()
        self.num_classes = num_classes
        self.rna_encoder = _rna_encoder(rna_embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        rna = x[0] if isinstance(x, (list, tuple)) else x
        return self.head(self.rna_encoder(rna))


def _create_baseline(classifier, pretrained, kwargs):
    if pretrained:
        raise ValueError("The baseline has no pretrained weights")
    # timm passes metadata keys that these classifiers do not use.
    for name in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(name, None)
    return classifier(**kwargs)


@register_model
def baseline_classifier(pretrained=False, **kwargs):
    """Create the registered baseline through timm.

    Args:
        pretrained: Must be False; no pretrained weights are supplied.
        **kwargs: Constructor options for BaselineClassifier.

    Returns:
        A BaselineClassifier instance.

    Raises:
        ValueError: If pretrained weights or unsupported fusion are requested.
    """
    return _create_baseline(BaselineClassifier, pretrained, kwargs)


@register_model
def wsi_only_classifier(pretrained=False, **kwargs):
    """Create the WSI-only baseline through timm (no pretrained weights)."""
    return _create_baseline(WSIOnlyClassifier, pretrained, kwargs)


@register_model
def rna_only_classifier(pretrained=False, **kwargs):
    """Create the RNA-only baseline through timm (no pretrained weights)."""
    return _create_baseline(RNAOnlyClassifier, pretrained, kwargs)


@register_model
def rna_cross_attention_classifier(pretrained=False, **kwargs):
    """Create the RNA-query cross-attention classifier through timm."""
    return _create_baseline(RNACrossAttentionClassifier, pretrained, kwargs)
