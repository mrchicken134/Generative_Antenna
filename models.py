import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    cDCGAN 生成器。

    输入:
    - 目标透射响应 M，对应 condition_dim
    - 随机噪声 R，对应 noise_dim

    输出:
    - 几何矩阵，通道数为 N（out_channels）
    """

    def __init__(self, condition_dim: int, noise_dim: int, out_channels: int):
        super().__init__()
        in_channels = condition_dim + noise_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, condition: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition, noise], dim=1)  # [B, M+R]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, M+R, 1, 1]
        return self.net(x)  # [B, N, 32, 32]


class Discriminator(nn.Module):
    """
    cDCGAN 判别器。

    输入:
    - 几何矩阵 geometry
    - 条件向量 condition（先映射为 2D 条件图）

    输出:
    - 真假概率标量
    """

    def __init__(self, condition_dim: int, geometry_channels: int, image_size: int = 32):
        super().__init__()
        self.image_size = image_size
        self.condition_map = nn.Linear(condition_dim, image_size * image_size)

        in_channels = geometry_channels + 1
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, geometry_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(geometry_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(geometry_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, geometry: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        b, _, h, w = geometry.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"几何矩阵尺寸应为 {self.image_size}x{self.image_size}，实际为 {h}x{w}。")

        cond = self.condition_map(condition).view(b, 1, self.image_size, self.image_size)
        x = torch.cat([geometry, cond], dim=1)
        x = self.features(x)
        return self.classifier(x)
