import torch
import torch.nn as nn

from metacell import PAPER_QUARTER_SIZE


def _generator_layers(in_channels: int, out_channels: int, image_size: int) -> nn.Sequential:
    if image_size == PAPER_QUARTER_SIZE:
        # 1 -> 3 -> 7 -> 15 -> 15, preserving the paper's 256/128/64/N channel plan.
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
        )

    if image_size == 32:
        return nn.Sequential(
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

    raise ValueError("Generator supports image_size=15 for the paper setup or image_size=32 for legacy runs.")


def _discriminator_layers(in_channels: int, geometry_channels: int, image_size: int) -> nn.Sequential:
    if image_size == PAPER_QUARTER_SIZE:
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, geometry_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(geometry_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    if image_size == 32:
        return nn.Sequential(
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

    raise ValueError("Discriminator supports image_size=15 for the paper setup or image_size=32 for legacy runs.")


class Generator(nn.Module):
    """
    cDCGAN generator from Liu et al. (2022).

    Inputs are the desired transmission response M and uniform random noise R.
    Output channels N represent geometry patterns in the compressed metacell.
    """

    def __init__(
        self,
        condition_dim: int,
        noise_dim: int,
        out_channels: int,
        image_size: int = PAPER_QUARTER_SIZE,
    ):
        super().__init__()
        self.image_size = image_size
        in_channels = condition_dim + noise_dim
        self.net = _generator_layers(in_channels, out_channels, image_size)

    def forward(self, condition: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition, noise], dim=1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


class Discriminator(nn.Module):
    """
    cDCGAN discriminator conditioned on transmission response and geometry.

    It follows the paper's five convolutional channel stages:
    64 -> 128 -> 256 -> 512 -> N, then a scalar real/fake output.
    """

    def __init__(
        self,
        condition_dim: int,
        geometry_channels: int,
        image_size: int = PAPER_QUARTER_SIZE,
    ):
        super().__init__()
        self.image_size = image_size
        self.condition_map = nn.Linear(condition_dim, image_size * image_size)

        in_channels = geometry_channels + 1
        self.features = _discriminator_layers(in_channels, geometry_channels, image_size)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(geometry_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, geometry: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        b, _, h, w = geometry.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"Geometry size should be {self.image_size}x{self.image_size}, got {h}x{w}.")

        cond = self.condition_map(condition).view(b, 1, self.image_size, self.image_size)
        x = torch.cat([geometry, cond], dim=1)
        x = self.features(x)
        return self.classifier(x)
