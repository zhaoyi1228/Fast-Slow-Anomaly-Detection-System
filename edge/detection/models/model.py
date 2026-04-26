"""
WideBranchNet Model
Jigsaw异常检测模型 - 3D卷积网络
从Unified_Jigsaw移植
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBranchNet(nn.Module):
    """WideBranchNet - 双分支3D卷积网络用于视频异常检测"""

    def __init__(self, time_length=7, num_classes=[127, 8], sample_num=None):
        """
        初始化模型

        Args:
            time_length: 时间维度长度（帧数）
            num_classes: 分类数 [spatial_classes, temporal_classes]
            sample_num: 采样帧数（兼容参数）
        """
        super(WideBranchNet, self).__init__()

        # 处理参数兼容
        if sample_num is not None:
            time_length = sample_num
        self.time_length = time_length
        self.num_classes = num_classes

        # 3D卷积层
        self.model = nn.Sequential(
            # 第一层
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            # 第二层
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            # 第三层
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(self.time_length, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
        )

        # 2D卷积层
        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
        )
        self.max2d = nn.MaxPool2d(2, 2)

        # 双分支分类器
        self.classifier_spatial = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[0])
        )
        self.classifier_temporal = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[1])
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [B, C, T, H, W]

        Returns:
            tuple: (spatial_logits, temporal_logits)
        """
        out = self.model(x)
        out = out.squeeze(2)  # 移除时间维度
        out = self.max2d(self.conv2d(out))
        out = out.view((out.size(0), -1))

        spatial_logits = self.classifier_spatial(out)
        temporal_logits = self.classifier_temporal(out)

        return spatial_logits, temporal_logits


if __name__ == '__main__':
    # 测试模型
    net = WideBranchNet(time_length=7, num_classes=[127, 8])
    x = torch.rand(2, 3, 7, 64, 64)
    out1, out2 = net(x)
    print(f"Spatial logits: {out1.shape}")
    print(f"Temporal logits: {out2.shape}")
    print(f"Total params: {sum(p.numel() for p in net.parameters())}")