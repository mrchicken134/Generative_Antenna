import argparse
from pathlib import Path

import numpy as np
import torch

from metacell import binarize_geometry, reconstruct_full_metacell
from models import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用训练好的 PK-cDCGAN 生成超构单元图案。")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt 文件路径。")
    parser.add_argument("--num_samples", type=int, default=8, help="未提供条件文件时的生成数量。")
    parser.add_argument("--condition_path", type=str, default="", help="可选条件向量 .npy 文件，形状为 [K, M]。")
    parser.add_argument("--output_path", type=str, default="generated_metacells.npz", help="输出 .npz 或 .npy 路径。")
    parser.add_argument("--threshold", type=float, default=0.5, help="导出二值图案时使用的阈值。")
    parser.add_argument("--seed", type=int, default=123, help="随机种子。")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt["args"]
    condition_dim = int(train_args["condition_dim"])
    noise_dim = int(train_args["noise_dim"])
    geometry_channels = int(train_args["geometry_channels"])
    image_size = int(train_args.get("image_size", 15))

    G = Generator(
        condition_dim=condition_dim,
        noise_dim=noise_dim,
        out_channels=geometry_channels,
        image_size=image_size,
    ).to(device)
    G.load_state_dict(ckpt["generator"])
    G.eval()

    if args.condition_path:
        cond_np = np.load(args.condition_path).astype(np.float32)
        if cond_np.ndim != 2 or cond_np.shape[1] != condition_dim:
            raise ValueError(f"条件文件形状应为 [K, {condition_dim}]，当前为 {cond_np.shape}。")
        cond = torch.from_numpy(cond_np).to(device)
    else:
        cond = torch.rand(args.num_samples, condition_dim, device=device)

    bsz = cond.size(0)
    z = torch.rand(bsz, noise_dim, device=device)

    with torch.no_grad():
        quarter_probability = G(cond, z).cpu().numpy()

    quarter_binary = binarize_geometry(quarter_probability, args.threshold)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".npy":
        np.save(output_path, quarter_probability)
    else:
        np.savez_compressed(
            output_path,
            quarter_probability=quarter_probability,
            quarter_binary=quarter_binary,
            full_layers=reconstruct_full_metacell(quarter_probability, threshold=args.threshold),
            conditions=cond.cpu().numpy(),
        )

    print(f"生成结果已保存到：{output_path.resolve()}")
    print(f"左上角概率图形状：{quarter_probability.shape}")


if __name__ == "__main__":
    main(parse_args())
