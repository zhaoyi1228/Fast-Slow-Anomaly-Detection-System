"""
Morphology operations for MAE model
From aed-mae project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Erosion2d(nn.Module):
    """
    Morphological erosion operation for 2D tensors.
    Used for post-processing anomaly detection results.
    """

    def __init__(self, m: int, n: int, k: int, soft_max: bool = True):
        """
        Args:
            m: input channels
            n: output channels
            k: kernel size (square kernel)
            soft_max: whether to use soft max approximation
        """
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.soft_max = soft_max

        # Create erosion kernel (structuring element)
        # For erosion, we use a kernel where all elements are 1
        weight = torch.ones(m, n, k, k)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, C, H, W)

        Returns:
            eroded tensor
        """
        if self.soft_max:
            # Soft erosion using min pooling approximation
            # Padding to maintain size
            pad = self.k // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')

            # Use unfolding to get local neighborhoods
            unfolded = F.unfold(x_padded, kernel_size=self.k)
            unfolded = unfolded.view(x.shape[0], self.m, self.k * self.k, -1)

            # Soft min (negative soft max)
            result = -F.softmax(-unfolded, dim=2).sum(dim=2)
            result = result.view(x.shape[0], self.n, x.shape[2], x.shape[3])
        else:
            # Hard erosion using min pooling
            pad = self.k // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')

            # Unfold and take minimum in each neighborhood
            unfolded = F.unfold(x_padded, kernel_size=self.k)
            unfolded = unfolded.view(x.shape[0], self.m, self.k * self.k, -1)
            result = unfolded.min(dim=2).values
            result = result.view(x.shape[0], self.n, x.shape[2], x.shape[3])

        return result


class Dilation2d(nn.Module):
    """
    Morphological dilation operation for 2D tensors.
    Used for post-processing anomaly detection results.
    """

    def __init__(self, m: int, n: int, k: int, soft_max: bool = True):
        """
        Args:
            m: input channels
            n: output channels
            k: kernel size (square kernel)
            soft_max: whether to use soft max approximation
        """
        super().__init__()
        self.m = m
        self.n = n
        self.k = k
        self.soft_max = soft_max

        # Create dilation kernel
        weight = torch.ones(m, n, k, k)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, C, H, W)

        Returns:
            dilated tensor
        """
        if self.soft_max:
            # Soft dilation using max pooling approximation
            pad = self.k // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')

            unfolded = F.unfold(x_padded, kernel_size=self.k)
            unfolded = unfolded.view(x.shape[0], self.m, self.k * self.k, -1)

            # Soft max for dilation
            result = F.softmax(unfolded, dim=2).sum(dim=2)
            result = result.view(x.shape[0], self.n, x.shape[2], x.shape[3])
        else:
            # Hard dilation using max pooling
            pad = self.k // 2
            x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')

            unfolded = F.unfold(x_padded, kernel_size=self.k)
            unfolded = unfolded.view(x.shape[0], self.m, self.k * self.k, -1)
            result = unfolded.max(dim=2).values
            result = result.view(x.shape[0], self.n, x.shape[2], x.shape[3])

        return result