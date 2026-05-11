import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MetacellDataset, SyntheticConfig, save_synthetic_npz
from metacell import PAPER_QUARTER_SIZE, binarize_geometry, reconstruct_full_metacell
from models import Discriminator, Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Liu et al. 2022 PK-cDCGAN metacell generator.")

    parser.add_argument("--dataset_npz", type=str, default="", help="HFSS/simulation dataset in npz format.")
    parser.add_argument(
        "--use_synthetic_if_missing",
        action="store_true",
        help="Generate paper-shaped synthetic data when --dataset_npz is not provided.",
    )
    parser.add_argument("--synthetic_path", type=str, default="synthetic_metacell.npz", help="Synthetic data path.")
    parser.add_argument("--synthetic_samples", type=int, default=6000, help="Synthetic sample count.")

    parser.add_argument("--condition_dim", type=int, default=64, help="M: desired transmission response dimension.")
    parser.add_argument("--noise_dim", type=int, default=32, help="R: uniform random noise dimension.")
    parser.add_argument(
        "--geometry_channels",
        type=int,
        default=2,
        help="N: 1 for identical triple layers, 2 for sandwich top/bottom + middle patterns.",
    )
    parser.add_argument("--image_size", type=int, default=PAPER_QUARTER_SIZE, help="Compressed geometry size.")
    parser.add_argument("--freq_min", type=float, default=8.0, help="Frequency range lower bound for metadata.")
    parser.add_argument("--freq_max", type=float, default=13.0, help="Frequency range upper bound for metadata.")
    parser.add_argument(
        "--no_paper_prior",
        action="store_true",
        help="Do not enforce diagonal symmetry and void rim while loading 15x15 data.",
    )

    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate used in the paper.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")

    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--save_every", type=int, default=20, help="Checkpoint interval.")
    parser.add_argument("--sample_every", type=int, default=10, help="Sample export interval.")
    parser.add_argument("--sample_count", type=int, default=16, help="Generated samples per export.")
    parser.add_argument("--sample_threshold", type=float, default=0.5, help="Threshold for binary metacell export.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset_path = args.dataset_npz
    if dataset_path and os.path.exists(dataset_path):
        dataset = MetacellDataset(
            dataset_path,
            image_size=args.image_size,
            enforce_paper_prior=not args.no_paper_prior,
        )
    else:
        if not args.use_synthetic_if_missing:
            raise FileNotFoundError("Dataset not found. Provide --dataset_npz or add --use_synthetic_if_missing.")

        cfg = SyntheticConfig(
            num_samples=args.synthetic_samples,
            condition_dim=args.condition_dim,
            geometry_channels=args.geometry_channels,
            image_size=args.image_size,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            seed=args.seed,
        )
        save_synthetic_npz(args.synthetic_path, cfg)
        dataset = MetacellDataset(args.synthetic_path, image_size=args.image_size)

    geo_shape = dataset.geometries.shape[1:]
    resp_dim = dataset.responses.shape[1]
    expected_geo_shape = (args.geometry_channels, args.image_size, args.image_size)
    if geo_shape != expected_geo_shape:
        raise ValueError(f"Dataset geometry shape should be {expected_geo_shape}, got {geo_shape}.")
    if resp_dim != args.condition_dim:
        raise ValueError(f"Dataset response dimension should be {args.condition_dim}, got {resp_dim}.")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )


def sample_uniform_noise(batch_size: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, noise_dim, device=device)


def save_generated_samples(
    generator: Generator,
    condition_dim: int,
    noise_dim: int,
    sample_count: int,
    epoch: int,
    out_dir: Path,
    device: torch.device,
    threshold: float,
) -> None:
    generator.eval()
    with torch.no_grad():
        cond = torch.rand(sample_count, condition_dim, device=device)
        z = sample_uniform_noise(sample_count, noise_dim, device)
        quarter_probability = generator(cond, z).cpu().numpy()
    generator.train()

    quarter_binary = binarize_geometry(quarter_probability, threshold)
    sample = {
        "quarter_probability": quarter_probability,
        "quarter_binary": quarter_binary,
        "full_layers": reconstruct_full_metacell(quarter_probability, threshold=threshold),
        "conditions": cond.cpu().numpy(),
    }
    np.savez_compressed(out_dir / f"samples_epoch_{epoch:04d}.npz", **sample)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(args)

    G = Generator(
        condition_dim=args.condition_dim,
        noise_dim=args.noise_dim,
        out_channels=args.geometry_channels,
        image_size=args.image_size,
    ).to(device)
    D = Discriminator(
        condition_dim=args.condition_dim,
        geometry_channels=args.geometry_channels,
        image_size=args.image_size,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    real_label = 1.0
    fake_label = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        g_epoch, d_epoch = 0.0, 0.0

        for real_geometry, real_response in loader:
            real_geometry = real_geometry.to(device)
            real_response = real_response.to(device)
            bsz = real_geometry.size(0)

            optimizer_D.zero_grad(set_to_none=True)
            real_targets = torch.full((bsz, 1), real_label, device=device)
            fake_targets = torch.full((bsz, 1), fake_label, device=device)

            pred_real = D(real_geometry, real_response)
            loss_real = criterion(pred_real, real_targets)

            z = sample_uniform_noise(bsz, args.noise_dim, device)
            fake_geometry = G(real_response, z)
            pred_fake = D(fake_geometry.detach(), real_response)
            loss_fake = criterion(pred_fake, fake_targets)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad(set_to_none=True)
            z2 = sample_uniform_noise(bsz, args.noise_dim, device)
            fake_geometry2 = G(real_response, z2)
            pred_fake_for_g = D(fake_geometry2, real_response)
            loss_G = criterion(pred_fake_for_g, real_targets)
            loss_G.backward()
            optimizer_G.step()

            g_epoch += loss_G.item()
            d_epoch += loss_D.item()

        g_epoch /= len(loader)
        d_epoch /= len(loader)
        history.append((epoch, d_epoch, g_epoch))
        print(f"[Epoch {epoch:04d}/{args.epochs}] D_loss={d_epoch:.6f} G_loss={g_epoch:.6f}")

        if epoch % args.sample_every == 0 or epoch == 1:
            save_generated_samples(
                generator=G,
                condition_dim=args.condition_dim,
                noise_dim=args.noise_dim,
                sample_count=args.sample_count,
                epoch=epoch,
                out_dir=out_dir,
                device=device,
                threshold=args.sample_threshold,
            )

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / f"ckpt_epoch_{epoch:04d}.pt")

    np.save(out_dir / "loss_history.npy", np.array(history, dtype=np.float32))
    print(f"Training complete. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    train(parse_args())
