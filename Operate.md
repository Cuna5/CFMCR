# EMRDM — Consistency Flow Matching 使用说明

本仓库当前只保留 **Consistency Flow Matching（CFM）** 作为 EMRDM 的加速优化方法。其他优化入口已移除；CFM 所需的时间映射、速度 scaling、损失和采样器均收敛到 `*_cfm.py` 文件中。

## 配置文件

| 数据集 | 配置文件 | 默认推理步数 |
|--------|----------|--------------|
| CUHK-CR1 | `configs/example_training/cuhk_cfm.yaml` | 1 |
| CUHK-CR2 | `configs/example_training/cuhkv2_cfm.yaml` | 1 |

## 训练

CFM 可以从零开始训练，不需要预训练 checkpoint：

```bash
python main.py --base configs/example_training/cuhk_cfm.yaml --train --enable_tf32
```

CUHK-CR2：

```bash
python main.py --base configs/example_training/cuhkv2_cfm.yaml --enable_tf32
```
如需从已有 CFM 权重继续训练，在 yaml 中填写：

```yaml
ckpt_path: "/path/to/cfm_pretrained.ckpt"
```

## 推理和评估

测试集评估：

```bash
python main.py \
    --base configs/example_training/cuhkv2_cfm.yaml \
    --test \
    --devices 0, \
    model.params.ckpt_path="/path/to/cfm_model.ckpt"\
    -t false
```

逐张预测并保存结果：

```bash
python main.py \
    --base configs/example_training/cuhkv2_cfm.yaml \
    --predict \
    --devices 0, \
    model.params.ckpt_path="/path/to/cfm_model.ckpt"\
    -t false
```

结果保存在 `logs/<experiment>/sample/`，包含 PNG、GeoTIFF 和 `metrics.csv`。`metrics.csv` 会记录每张图像的 PSNR / SSIM / RMSE 等指标。

## 画质指标优化

当前 CFM 针对 1 步推理画质做了四处优化：

| 参数 / 实现 | 当前设置 | 作用 |
|-------------|----------|------|
| `clean_endpoint_loss_weight` | `1.0` | 直接约束 `fθ=x+σv` 接近 `x_clean`，让训练目标更贴近 PSNR / SSIM / RMSE 的评估端点 |
| `start_pair_prob` | `0.35` | 提高采样 `σ≈1` 起点段的概率，强化真正 1 步推理时的 `μ -> x_clean` 能力 |
| `loss_fn_config.params.num_steps` | `40` | 比原先更细的 σ pair 覆盖，降低相邻时间点过粗带来的 teacher 目标误差 |
| `endpoint_loss_weight` / `consistency_loss_weight` | `0.5 / 0.5` | 降低 EMA teacher 自一致性项的主导性，避免早期或滞后 teacher 压过真实 clean 监督 |

默认 `sampler_config.params.num_steps=1`，用于最快推理。若某个数据集上 1 步结果仍偏平滑，可以在评估时临时提高步数：

```bash
model.params.sampler_config.params.num_steps=3
```

这会使用 CFM 采样器内置的 Euler 多步路径，通常能换取更好的 PSNR / SSIM，但推理耗时会随步数增加。

## 关键参数

| 参数路径 | 默认值 | 说明 |
|----------|--------|------|
| `model.base_learning_rate` | `1e-4` | CFM 从零训练初始学习率 |
| `model.params.teacher_ema_decay` | `0.9999` | EMA teacher 衰减率 |
| `model.params.loss_fn_config.params.loss_type` | `"charbonnier"` | 默认使用 Charbonnier endpoint / velocity 损失 |
| `model.params.loss_fn_config.params.charbonnier_eps` | `1.0e-3` | Charbonnier 平滑项 ε，公式为 `sqrt(diff^2 + ε^2)` |
| `model.params.loss_fn_config.params.velocity_anchor_loss_weight` | `1.0` | 速度锚点损失，约束 `vθ ≈ x_clean - μ` |
| `model.params.loss_fn_config.params.clean_endpoint_loss_weight` | `1.0` | clean endpoint 监督，优先服务画质指标 |
| `model.params.loss_fn_config.params.start_pair_prob` | `0.35` | 起点段过采样概率 |
| `model.params.loss_fn_config.params.consistency_warmup_steps` | `2000` | 前期线性引入 teacher 一致性项 |
| `model.params.sampler_config.params.num_steps` | `1` | 推理步数；设为 `3` 或 `4` 可做质量/速度折中 |

## 数据集路径

修改 yaml 中的数据路径：

```yaml
data:
  params:
    train:
      params:
        datasets_dir: "/your/path/to/CUHK-CR1"
        nir_datasets_dir: "/your/path/to/nir/CUHK-CR1"
```

## 耗时统计

如需比较速度，打开：

```yaml
model:
  params:
    count_sample_time: True
    count_train_time: True
```

`sample_time` 会覆盖 validation / test / predict 的 sample + decode 阶段；`train_time` 和 `train_time_avg` 会记录训练 batch 耗时，不包含 dataloader 取数时间。
