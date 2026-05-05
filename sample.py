import argparse
from pathlib import Path

import numpy as np
import torch

from models import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的 cDCGAN 生成器进行几何矩阵采样。")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt 文件路径。")
    parser.add_argument("--num_samples", type=int, default=8, help="当未提供条件文件时的采样数量。")
    parser.add_argument("--condition_path", type=str, default="", help="可选，条件向量 .npy 文件，形状应为 [K, M]。")
    parser.add_argument("--output_path", type=str, default="generated_geometries.npy", help="生成结果输出路径。")
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

    G = Generator(
        condition_dim=condition_dim,
        noise_dim=noise_dim,
        out_channels=geometry_channels,
    ).to(device)
    G.load_state_dict(ckpt["generator"])
    G.eval()

    if args.condition_path:
        cond_np = np.load(args.condition_path).astype(np.float32)
        if cond_np.ndim != 2 or cond_np.shape[1] != condition_dim:
            raise ValueError(f"条件文件形状应为 [K, {condition_dim}]，实际为 {cond_np.shape}。")
        cond = torch.from_numpy(cond_np).to(device)
    else:
        cond = torch.rand(args.num_samples, condition_dim, device=device)

    bsz = cond.size(0)
    z = torch.rand(bsz, noise_dim, device=device)  # 均匀分布噪声

    with torch.no_grad():
        generated = G(cond, z).cpu().numpy()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, generated)
    print(f"生成几何矩阵已保存到: {output_path.resolve()}")
    print(f"输出形状: {generated.shape}")


if __name__ == "__main__":
    main(parse_args())
