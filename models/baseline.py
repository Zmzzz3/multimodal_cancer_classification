"""Small ABMIL and MLP baseline for paired tumour classification."""

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
        rna_layers = []
        for in_features in (rna_embed_dim, embed_dim):
            rna_layers.extend([
                nn.Linear(in_features, embed_dim), nn.LayerNorm(embed_dim),
                nn.ReLU(), nn.Dropout(0.2),
            ])
        rna_layers.append(nn.Linear(embed_dim, embed_dim))
        self.rna_encoder = nn.Sequential(*rna_layers)
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
    if pretrained:
        raise ValueError("The baseline has no pretrained weights")
    # timm passes these metadata keys even though this model does not use them.
    for name in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(name, None)
    return BaselineClassifier(**kwargs)
