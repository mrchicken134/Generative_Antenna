# Liu et al. 2022 PK-cDCGAN 复现代码

本项目复现论文 **Prior-Knowledge-Guided Deep-Learning-Enabled Synthesis for Broadband and Large Phase Shift Range Metacells in Metalens Antenna** 中的条件 DCGAN 超构单元生成流程。

代码重点复现论文可从文本中确定的部分：

- 生成器输入：目标透射响应向量 `M` 与均匀随机噪声 `R` 拼接。
- 生成器结构：反卷积通道 `256 -> 128 -> 64 -> N`。
- 判别器结构：卷积通道 `64 -> 128 -> 256 -> 512 -> N`，最终输出真假标量。
- 训练目标：标准 GAN minimax，用 BCE 损失实现。
- 优化器：Adam，默认学习率 `2e-4`。
- 先验几何：30x30 像素超构单元压缩为 15x15 上左象限，外圈像素置 0，并施加水平、垂直和对角镜像对称。
- 多层结构：`N=1` 表示三层相同图案；`N=2` 表示三明治结构，即上下层相同、中间层不同。

真实论文效果需要 HFSS 或其他全波仿真生成的训练数据。仓库中的合成数据只用于检查训练、保存和采样流程是否可跑通，不能替代电磁仿真数据。

## 文件说明

- `models.py`：cDCGAN 生成器和判别器。
- `dataset.py`：论文几何先验数据读取，以及可跑通流程的合成数据生成。
- `metacell.py`：15x15 压缩图案到 30x30 三层超构单元的重构工具。
- `train.py`：训练入口。
- `sample.py`：checkpoint 采样入口。
- `detail.txt`：论文核心网络配置摘要。

## 数据格式

训练数据使用 `.npz`，至少包含：

- `geometries`：形状 `[num_samples, N, H, W]`。
- `responses`：形状 `[num_samples, M]`。

推荐直接保存论文训练表示：

- `H=W=15`：上左象限压缩图案。
- `N=1`：三层同图案。
- `N=2`：上下层共享图案 + 中间层图案。

如果输入 `H=W=30`，`MetacellDataset` 会在 `image_size=15` 时自动截取上左 15x15 象限。真实数据中的 `responses` 应来自全波仿真的透射响应，或实验中定义的目标透射响应采样。

## 快速开始

安装依赖：

```bash
python -m pip install -r requirements.txt
```

用合成数据做流程自检：

```bash
python train.py --use_synthetic_if_missing --epochs 5 --batch_size 32
```

使用真实 HFSS 数据训练三明治结构：

```bash
python train.py --dataset_npz your_hfss_dataset.npz --condition_dim 64 --noise_dim 32 --geometry_channels 2 --image_size 15 --epochs 200
```

训练三层完全相同图案的版本：

```bash
python train.py --dataset_npz your_hfss_dataset.npz --geometry_channels 1 --image_size 15 --epochs 200
```

从 checkpoint 生成超构单元：

```bash
python sample.py --ckpt outputs/ckpt_epoch_0200.pt --num_samples 8 --output_path outputs/generated_metacells.npz
```

使用自定义目标透射响应采样：

```bash
python sample.py --ckpt outputs/ckpt_epoch_0200.pt --condition_path target_responses.npy --output_path outputs/generated_from_condition.npz
```

采样输出 `.npz` 包含：

- `quarter_probability`：生成器输出的 `N x 15 x 15` 概率图。
- `quarter_binary`：阈值化后的 `N x 15 x 15` 二值图。
- `full_layers`：重构后的三层 `3 x 30 x 30` 超构单元。
- `conditions`：对应的条件响应向量。

## 论文参数映射

- `M`：`--condition_dim`
- `R`：`--noise_dim`
- `N`：`--geometry_channels`
- 目标频段：`--freq_min`、`--freq_max`，用于数据/实验元信息
- 阈值化：`--sample_threshold` 或 `sample.py --threshold`

## 注意事项

论文中的高性能结果依赖 “全波仿真数据 + 先验筛选 + 训练后再次仿真验证”。本仓库复现的是可训练的 PK-cDCGAN 生成框架和几何先验编码，不包含 HFSS 自动建模、仿真调用或金属透镜整机仿真流程。
