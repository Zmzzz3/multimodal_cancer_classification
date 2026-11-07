"""Multimodal and unimodal ABMIL/MLP tumour classifiers."""

import torch
from torch import nn
from timm.models import register_model


class AttentionPool(nn.Module):
    """Project patch features and pool each bag with gated attention.

    Args:
        input_dim: Number of features per WSI patch.
        width: Width of the projected patches and pooled embedding.
    """

    def __init__(self, input_dim, width):
        super().__init__()
        layers = []
        for in_features in (input_dim, width):
            layers.extend([
                nn.Linear(in_features, width), nn.LayerNorm(width),
                nn.GELU(), nn.Dropout(0.1),
            ])
        self.project = nn.Sequential(*layers)
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
