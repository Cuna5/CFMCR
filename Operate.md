# EMRDM — Consistency Distillation 使用说明

## 环境要求

与原始 EMRDM 相同，参考 `requirements.txt` 安装依赖：

```bash
pip install -r requirements.txt
```

---

## 训练

### 两阶段推荐流程

#### 第一阶段：训练基础 EMRDM 模型

使用原始配置文件先训练一个完整的 EMRDM teacher 模型：

```bash
python main.py \
    --base configs/example_training/cuhk.yaml \
    --train \
    --devices 0,1 \
    lightning.trainer.max_epochs=500
```

训练完成后在 `logs/` 目录下找到 checkpoint 路径（如 `logs/xxx/checkpoints/last.ckpt`）。

#### 第二阶段：一致性蒸馏微调（1步去云）

打开 `configs/example_training/cuhk_consistency.yaml`，取消注释并填写第一阶段的 checkpoint 路径：

```yaml
# 约第 29 行
ckpt_path: "/path/to/emrdm_pretrained.ckpt"
```

然后启动蒸馏训练：

```bash
python main.py \
    --base configs/example_training/cuhk_consistency.yaml \
    --train \
    --devices 0,1 \
    lightning.trainer.max_epochs=200
```

> **提示**：如果 GPU 资源有限，可先用小批量验证流程：
> ```bash
> python main.py \
>     --base configs/example_training/cuhk_consistency.yaml \
>     --train \
>     --devices 0, \
>     data.params.batch_size=1 \
>     lightning.trainer.max_epochs=5
> ```

---

## 推理（1步去云）

蒸馏训练完成后，模型推理只需 **1 次** 网络前向即可输出无云图像。

### 使用测试集评估

```bash
python main.py \
    --base configs/example_training/cuhk_consistency.yaml \
    --test \
    --devices 0, \
    model.params.ckpt_path="/path/to/consistency_model.ckpt"
```

### 使用 predict 模式逐张预测并保存结果

```bash
python main.py \
    --base configs/example_training/cuhk_consistency.yaml \
    --predict \
    --devices 0, \
    model.params.ckpt_path="/path/to/consistency_model.ckpt"
```

结果保存在 `logs/<experiment>/sample/` 目录下，包含：
- `<image_name>.png`：RGB 预测结果
- `<image_name>.tif`：多波段 GeoTIFF 预测结果
- `<image_name>_target.png`：目标无云图像（RGB）
- `<image_name>_mu.png`：含云输入图像（RGB）
- `metrics.csv`：每张图像的 PSNR / SSIM / RMSE 等指标

---

## 配置文件关键参数说明

`configs/example_training/cuhk_consistency.yaml` 中可调整的关键参数：

| 参数路径 | 默认值 | 说明 |
|----------|--------|------|
| `model.base_learning_rate` | `2e-5` | 蒸馏阶段学习率，建议比 base 训练低 5–10× |
| `model.params.teacher_ema_decay` | `0.9999` | Teacher EMA 衰减率，越大 teacher 越"滞后" |
| `model.params.loss_fn_config.params.num_steps` | `18` | 训练时采样 σ 对的时间表密度，更大值覆盖更多噪声水平 |
| `model.params.loss_fn_config.params.loss_type` | `"l2"` | 一致性损失类型，可选 `"l1"` |
| `model.params.sampler_config.params.num_steps` | `4` | 仅用于确定 σ_max，推理实际只走 1 步 |
| `lightning.trainer.max_epochs` | `200` | 蒸馏训练总 epoch 数 |
| `data.params.batch_size` | `4` | 批大小，根据 GPU 显存调整 |

---

## 数据集路径配置

修改配置文件中的数据集路径以适配本地环境：

```yaml
data:
  params:
    train:
      params:
        datasets_dir: "/your/path/to/CUHK-CR1"
        nir_datasets_dir: "/your/path/to/nir/CUHK-CR1"
```

---

## 多时相（Sen2_MTC）扩展

如需对多时相数据集进行一致性蒸馏，参照 `cuhk_consistency.yaml` 的改动方式修改 `sen2_mtc_new.yaml`：

1. 将 `model.target` 改为 `ConsistencyResidualDiffusionEngine`
2. 将 `loss_fn_config.target` 改为 `ConsistencyResidualDiffusionLoss`
3. 将 `sampler_config.target` 改为 `ConsistencyResidualSampler`

> **注意**：多时相 engine 当前为 `TemporalResidualDiffusionEngine`，继承链与 `ResidualDiffusionEngine` 相同，`ConsistencyResidualDiffusionEngine` 可直接替换使用。

---

## 与原始 EMRDM 性能对比

| 模式 | 推理步数 | 相对速度 | 备注 |
|------|----------|----------|------|
| 原始 EMRDM（Euler） | 4 步 | 1× | 基线 |
| Consistency Distillation | **1 步** | **~4× 加速** | 本次改进 |

---

## 常见问题

**Q: 蒸馏训练时 loss 震荡严重怎么办？**  
A: 降低 `base_learning_rate`（尝试 `5e-6`），或增大 `teacher_ema_decay`（尝试 `0.99999`）使 teacher 更稳定。

**Q: 1步推理结果质量不如 4步怎么办？**  
A: 可在推理时使用 `ResidualEulerEDMSampler`（2步）配合蒸馏后的权重，往往能在速度与质量间取得更好平衡。

**Q: 如何从零开始（不加载预训练）训练一致性模型？**  
A: 删除或注释掉 `ckpt_path`，适当增大 `max_epochs`（建议 500+）并降低学习率至 `1e-5`。效果通常不如两阶段方法，但无需预训练模型。

---

---

# Flow Matching 使用说明

## 概述

Flow Matching（FM）以直线 OT 轨迹替代扩散过程，**从零开始训练**（无需预训练 checkpoint），训练更稳定、收敛更快。配置文件：

| 数据集 | 配置文件 |
|--------|----------|
| CUHK-CR1（单时相） | `configs/example_training/cuhk_fm.yaml` |
| CUHK-CR2（多时相） | `configs/example_training/cuhkv2_fm.yaml` |

---

## 训练

### 单时相（CUHK-CR1）

FM **无需**预训练，直接从零开始训练：

```bash
python main.py \
    --base configs/example_training/cuhk_fm.yaml \
    --train \
    --devices 0,1 \
    lightning.trainer.max_epochs=500
```

### 多时相（CUHK-CR2）

```bash
python main.py \
    --base configs/example_training/cuhkv2_fm.yaml \
    --train \
    --devices 0,1 \
    lightning.trainer.max_epochs=2000
```

> **提示**：如需加快收敛，可加载 EMRDM 预训练权重进行微调（不强制要求）：
> ```yaml
> # 在 yaml 中取消注释并填写路径
> ckpt_path: "/path/to/emrdm_pretrained.ckpt"
> ```

---

## 推理

### 测试集评估

```bash
python main.py \
    --base configs/example_training/cuhk_fm.yaml \
    --test \
    --devices 0, \
    model.params.ckpt_path="/path/to/fm_model.ckpt"
```

### 逐张预测并保存结果

```bash
python main.py \
    --base configs/example_training/cuhk_fm.yaml \
    --predict \
    --devices 0, \
    model.params.ckpt_path="/path/to/fm_model.ckpt"
```

输出目录结构与 CD 相同（`logs/<experiment>/sample/`），包含 PNG、GeoTIFF 及 `metrics.csv`。

### 调整推理步数

FM 使用 Euler 积分，步数越多质量越好，但速度越慢。在命令行覆盖：

```bash
model.params.sampler_config.params.num_steps=20
```

推荐范围：`5`（快速）到 `20`（高质量）。

---

## 配置文件关键参数说明

`configs/example_training/cuhk_fm.yaml` 中可调整的关键参数：

| 参数路径 | 默认值 | 说明 |
|----------|--------|------|
| `model.base_learning_rate` | `1e-4` | 初始学习率（FM 从零训练可用较大值） |
| `model.params.loss_fn_config.params.t_min` | `0.001` | 训练时最小时间步（避免退化到 $t=0$） |
| `model.params.loss_fn_config.params.t_max` | `0.999` | 训练时最大时间步 |
| `model.params.loss_fn_config.params.loss_type` | `"l2"` | 速度场损失类型，可选 `"l1"` |
| `model.params.sampler_config.params.num_steps` | `10` | 推理 Euler 步数 |
| `model.params.network_config.params.sigma_max` | `0.999` | 最大噪声水平（对应 $t_{min}$） |
| `model.params.network_config.params.sigma_min` | `0.001` | 最小噪声水平（对应 $t_{max}$） |
| `lightning.trainer.max_epochs` | `500` | 训练总 epoch 数 |
| `data.params.batch_size` | `4` | 批大小，根据显存调整 |

---

## 数据集路径配置

与 CD 配置相同，修改 yaml 中的路径：

```yaml
data:
  params:
    train:
      params:
        datasets_dir: "/your/path/to/CUHK-CR1"
        nir_datasets_dir: "/your/path/to/nir/CUHK-CR1"
```

---

## 三种方法完整对比

| 模式 | 配置文件 | 推理步数 | 需要预训练 | 主要优势 |
|------|----------|----------|-----------|----------|
| 原始 EMRDM | `cuhk.yaml` | 4–5 步 | 否 | 精度高、经过验证 |
| Consistency Distillation | `cuhk_consistency.yaml` | **1 步** | **是** | 推理极速（~4× 加速） |
| Flow Matching | `cuhk_fm.yaml` | 5–20 步（可调） | 否 | 训练稳定、收敛快 |

---

## 常见问题

**Q: FM 训练 loss 不下降怎么办？**  
A: 检查 `t_min`/`t_max` 是否正确（应避免 $t=0$ 或 $t=1$ 的端点）；尝试降低学习率至 `5e-5`。

**Q: FM 推理结果模糊怎么办？**  
A: 增加推理步数（`num_steps=20`），或尝试使用 Heun 积分（将来可在 `sampling_fm.py` 中扩展 `FlowMatchingResidualSampler`）。

**Q: FM 和 CD 可以结合使用吗？**  
A: 可以。先用 FM 训练一个基础模型（快速收敛），再以此为 teacher 做一致性蒸馏，得到 1 步 FM 模型。将 `cuhk_consistency.yaml` 的 `ckpt_path` 指向 FM checkpoint 即可。

**Q: 如何切换 CUHK-CR2 数据集？**  
A: 使用 `cuhkv2_fm.yaml` 替换 `cuhk_fm.yaml`，该配置已调整 `batch_size=1` 和 `max_epochs=2000` 以适配多时相场景。

---

---

# Consistency Flow Matching 使用说明

## 概述

Consistency Flow Matching（CFM）综合了 FM 的直线 OT 路径和 CD 的一步推理能力：
- **从零训练**（无需预训练 checkpoint）
- **1 步推理**（默认 `num_steps=1`）
- teacher 仅需 1 次前向（比 CD 更高效）

| 数据集 | 配置文件 |
|--------|----------|
| CUHK-CR1（单时相） | `configs/example_training/cuhk_cfm.yaml` |
| CUHK-CR2（多时相） | `configs/example_training/cuhkv2_cfm.yaml` |

---

## 训练

### 单时相（CUHK-CR1）

```bash
python main.py \
    --base configs/example_training/cuhk_cfm.yaml \
    --train \
    --devices 0,1 \
    lightning.trainer.max_epochs=500
```

### 多时相（CUHK-CR2）

```bash
python main.py \
    --base configs/example_training/cuhkv2_cfm.yaml \
    --train \
    --devices 0,1 \
    lightning.trainer.max_epochs=2000
```

> **提示**：可选择以 FM checkpoint 为起点进行微调，加快 CFM 收敛：
> ```yaml
> # 在 yaml 中取消注释并填写路径
> ckpt_path: "/path/to/fm_pretrained.ckpt"
> ```

---

## 推理

### 1 步去云（默认模式）

```bash
python main.py \
    --base configs/example_training/cuhk_cfm.yaml \
    --test \
    --devices 0, \
    model.params.ckpt_path="/path/to/cfm_model.ckpt"
```

### 多步推理（提升质量）

在命令行将 `num_steps` 设为大于 1 的值，此时采样器自动切换为 Euler ODE 循环：

```bash
python main.py \
    --base configs/example_training/cuhk_cfm.yaml \
    --test \
    --devices 0, \
    model.params.ckpt_path="/path/to/cfm_model.ckpt" \
    model.params.sampler_config.params.num_steps=5
```

### 预测并保存结果

```bash
python main.py \
    --base configs/example_training/cuhk_cfm.yaml \
    --predict \
    --devices 0, \
    model.params.ckpt_path="/path/to/cfm_model.ckpt"
```

---

## 配置文件关键参数说明

| 参数路径 | 默认值 | 说明 |
|----------|--------|------|
| `model.base_learning_rate` | `1e-4` | 初始学习率（从零训练可用较大值） |
| `model.params.teacher_ema_decay` | `0.9999` | Teacher EMA 衰减率 |
| `model.params.loss_fn_config.params.num_steps` | `18` | 训练时 σ 时间表密度，越大覆盖越多噪声水平 |
| `model.params.loss_fn_config.params.loss_type` | `"l2"` | 速度一致性损失类型，可选 `"l1"` |
| `model.params.sampler_config.params.num_steps` | `1` | 推理步数（`1` = 单步；`>1` = Euler 多步） |
| `lightning.trainer.max_epochs` | `500` | 训练总 epoch 数 |
| `data.params.batch_size` | `4` | 批大小，根据显存调整 |

---

## 数据集路径配置

与其他方法相同，修改 yaml 中的路径：

```yaml
data:
  params:
    train:
      params:
        datasets_dir: "/your/path/to/CUHK-CR1"
        nir_datasets_dir: "/your/path/to/nir/CUHK-CR1"
```

---

## 四种方法完整对比

| 模式 | 配置文件 | 推理步数 | 需预训练 | 主要优势 |
|------|----------|----------|---------|----------|
| 原始 EMRDM | `cuhk.yaml` | 4–5 步 | 否 | 精度最高、经过充分验证 |
| Consistency Distillation | `cuhk_consistency.yaml` | **1 步** | **是** | 推理极速（~4× 加速） |
| Flow Matching | `cuhk_fm.yaml` | 4–10 步（可调） | 否 | 训练稳定、收敛快 |
| **Consistency Flow Matching** | `cuhk_cfm.yaml` | **1 步**（可调） | **否** | **1步推理 + 无需预训练** |

---

## 常见问题

**Q: CFM 和 CD 哪个 1 步效果更好？**  
A: 取决于数据集。CFM 训练路径更直（无额外噪声），理论上一致性约束更稳定；CD 利用了已收敛的 EMRDM 先验，初期收敛更快。建议两者都尝试。

**Q: CFM 与 FM 的区别是什么？**  
A: 两者都使用直线 OT 路径，但 FM 需要多步推理（4–10 步），CFM 通过一致性约束实现 1 步推理。代价是需要额外维护一个 EMA teacher 模型。

**Q: `num_steps=1` 和 `num_steps=5` 推理结果差多少？**  
A: 经验上 CFM 1 步已接近多步质量（这是一致性训练的优势），但如遇质量不足可先尝试 `num_steps=3` 作为折中。

**Q: 如何切换 CUHK-CR2 数据集？**  
A: 使用 `cuhkv2_cfm.yaml` 替换 `cuhk_cfm.yaml`，该配置已调整 `batch_size=1`、`max_epochs=2000` 以适配多时相场景。

---

---

# 推理性能统计

## 概述

代码内置 `count_sample_time` 和 `count_train_time` 计时机制（CUDA Event 精确计时），可对比各改进方向的推理/训练速度。指标会实时写入 TensorBoard 日志和 `metrics.csv`。

---

## 开启方式

在任意 YAML 配置文件中添加：

```yaml
model:
  params:
    count_sample_time: True   # 统计推理耗时（ms/batch），写入 sample_time 指标
    count_train_time: True    # 统计训练耗时（ms/batch），写入 train_time 指标
```

也可以在命令行直接覆盖，无需修改 YAML：

```bash
python main.py \
    --base configs/example_training/cuhk_cfm.yaml \
    --test \
    --devices 0, \
    model.params.ckpt_path="/path/to/model.ckpt" \
    model.params.count_sample_time=True
```

---

## 查看结果

### TensorBoard

```bash
tensorboard --logdir logs/
```

在 Scalars 面板中查看 `sample_time`（每 step 的推理耗时，单位毫秒）和 `train_time`（每 step 的训练耗时）。

### CSV 日志

```bash
grep "sample_time" logs/<experiment>/metrics.csv
```

---

## 对比不同方法的推理性能

在相同 GPU 和相同测试集上，分别对各方法运行 `--test`：

```bash
# 原始 EMRDM（4步，基线）
python main.py --base configs/example_training/cuhk.yaml --test \
    --devices 0, model.params.ckpt_path="..." model.params.count_sample_time=True

# Consistency Distillation（1步）
python main.py --base configs/example_training/cuhk_consistency.yaml --test \
    --devices 0, model.params.ckpt_path="..." model.params.count_sample_time=True

# Flow Matching（10步）
python main.py --base configs/example_training/cuhk_fm.yaml --test \
    --devices 0, model.params.ckpt_path="..." model.params.count_sample_time=True

# Consistency Flow Matching（1步）
python main.py --base configs/example_training/cuhk_cfm.yaml --test \
    --devices 0, model.params.ckpt_path="..." model.params.count_sample_time=True
```

读取各实验的 `sample_time` 均值，即可得到推理速度对比表：

| 方法 | 推理步数 | 预期相对耗时 |
|------|----------|-------------|
| 原始 EMRDM | 4 步 | 1× （基线） |
| CD | **1 步** | ~25% |
| FM (10步) | 10 步 | ~250% |
| CFM | **1 步** | ~25% |

---

## 对比训练效率

开启 `count_train_time=True` 后，`train_time`（ms/batch）写入日志，可用于验证 CFM 的 teacher 计算量是否比 CD 更低：

- CD teacher：EMRDM Euler 步骤 + **2 次** teacher 前向
- CFM teacher：OT 公式直接计算 + **1 次** teacher 前向

理论上在相同 batch size 和网络结构下，CFM 每步训练应快于 CD。
