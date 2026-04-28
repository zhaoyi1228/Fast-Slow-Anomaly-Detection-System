"""
Model factory for MAE-CvT models
From aed-mae project
"""

from functools import partial
from torch import nn
from .mae_cvt import MaskedAutoencoderCvT


def mae_cvt_patch16(**kwargs):
    """
    Create MAE-CvT model with patch size 16
    Args:
        **kwargs: Additional model parameters (img_size, norm_pix_loss, etc.)
    Returns:
        MaskedAutoencoderCvT model instance
    """
    model = MaskedAutoencoderCvT(
        patch_size=16, embed_dim=256, depth=3, num_heads=4,
        decoder_embed_dim=128, decoder_depth=3, decoder_num_heads=4,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_cvt_patch8(**kwargs):
    """
    Create MAE-CvT model with patch size 8
    Args:
        **kwargs: Additional model parameters (img_size, norm_pix_loss, etc.)
    Returns:
        MaskedAutoencoderCvT model instance
    """
    model = MaskedAutoencoderCvT(
        patch_size=8, embed_dim=256, depth=3, num_heads=4,
        decoder_embed_dim=128, decoder_depth=3, decoder_num_heads=4,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model