# cDCGAN 复现项目（超构单元几何生成）

本项目依据 `detail.txt` 中的要求复现 cDCGAN：

- 生成器（Generator）：反卷积 4 层，通道数 `256 -> 128 -> 64 -> N`
- 判别器（Discriminator）：卷积主干通道 `64 -> 128 -> 256 -> 512 -> N`，最终输出真假标量
- 条件输入：
  - 生成器输入为目标透射响应向量 `M` 与随机噪声 `R` 的拼接
  - 噪声服从均匀分布（uniform）
- 训练目标：标准 GAN minimax（代码中用 BCE 实现）
- 优化器：Adam，默认学习率 `2e-4`

## 项目文件

- `models.py`：生成器与判别器定义
- `dataset.py`：真实数据读取 + 合成数据生成
- `train.py`：训练入口
- `sample.py`：推理/采样入口
- `requirements.txt`：依赖版本

## 数据格式

训练数据使用 `.npz`，包含：

- `geometries`：形状 `[num_samples, N, H, W]`，取值建议在 `[0, 1]`
- `responses`：形状 `[num_samples, M]`

## 快速开始

安装依赖：

```bash
python -m pip install -r requirements.txt
```

使用真实数据训练：

```bash
python train.py --dataset_npz your_dataset.npz --condition_dim 64 --noise_dim 32 --geometry_channels 4 --image_size 32 --epochs 200
```

使用合成数据训练（流程自检）：

```bash
python train.py --use_synthetic_if_missing --condition_dim 64 --noise_dim 32 --geometry_channels 4 --image_size 32 --epochs 50
```

从 checkpoint 采样几何矩阵：

```bash
python sample.py --ckpt outputs/ckpt_epoch_0200.pt --num_samples 8 --output_path outputs/generated.npy
```

使用自定义目标响应向量采样（`.npy` 形状 `[K, M]`）：

```bash
python sample.py --ckpt outputs/ckpt_epoch_0200.pt --condition_path your_conditions.npy --output_path outputs/generated_from_condition.npy
```

## 已同步环境版本

- Python：`3.10.19`
- NumPy：`2.2.6`
- PyTorch：`2.10.0+cu130`
- TorchVision：`0.25.0+cu130`

## 参数与文档符号映射

- `M` 对应 `--condition_dim`
- `R` 对应 `--noise_dim`
- `N` 对应 `--geometry_channels`
- 目标频段对应 `--freq_min`、`--freq_max`（用于实验配置记录）
- 目标透射响应由数据集中的 `responses` 提供（或由合成数据生成）
