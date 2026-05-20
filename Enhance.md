# EMRDM 改进记录 — Consistency Flow Matching

本轮整理后，仓库只保留 **Consistency Flow Matching（CFM）** 作为 EMRDM 的优化方法。其他优化配置、代码入口与文档内容已经删除；CFM 需要的时间映射、速度 scaling、损失函数和采样器都放在 `*_cfm.py` 中维护。

## 核心目标

CFM 在 EMRDM 框架中使用直线 OT 路径：

$$x_t=(1-t)\mu+t x_{clean}$$

网络预测速度场：

$$v_\theta(x_t,t,\mu)\approx x_{clean}-\mu$$

并通过端点映射完成一步去云：

$$f_\theta(t,x_t,\mu)=x_t+(1-t)v_\theta(x_t,t,\mu)$$

推理时从含云图像开始：

$$x_{pred}=\mu+\sigma_{max}v_\theta(\mu,\sigma_{max},\mu)$$

默认 `sigma_max=1.0`、`sampler.num_steps=1`，因此只需要一次网络前向。

## 当前 CFM 损失

`sgm/modules/diffusionmodules/loss_cfm.py` 中的 `ConsistencyFlowMatchingLoss` 包含四个部分：

| 损失项 | 公式 | 作用 |
|--------|------|------|
| teacher 端点一致性 | `||f_student - f_teacher||` | 让同一 OT 轨迹上的端点预测保持一致 |
| teacher 速度一致性 | `||v_student - v_teacher||` | 让速度场沿轨迹稳定 |
| 速度锚点 | `||v_student - (x_clean - μ)||` | 防止自一致但方向错误 |
| clean endpoint 监督 | `||f_student - x_clean||` | 直接优化 1 步输出端点，服务 PSNR / SSIM / RMSE |

当前 CFM 配置还对监督项加入了云区加权：从 batch 中读取 `M` mask，对 `clean_endpoint_loss` 加权，并通过 `cloud_weight_velocity_anchor=True` 同步加权速度锚点。权重图按每个样本的均值归一化，因此强调云区的同时不会显著改变整体 loss 尺度。

当前组合：

$$
\mathcal{L}=
r\lambda_f\mathcal{L}_{end}
+r\lambda_v\mathcal{L}_{vel}
+\lambda_a\mathcal{L}_{anchor}
+\lambda_c\mathcal{L}_{clean}
$$

其中 `r` 是 `consistency_warmup_steps` 控制的线性 warmup。前期先依靠真实监督学习正确速度，再逐步增强 EMA teacher 的一致性约束。

## 画质指标优化

为了降低 1 步 CFM 相比原始多步 EMRDM 的画质损失，本轮做了以下处理：

1. 新增 `clean_endpoint_loss_weight`
   直接监督 `fθ=x+σv` 对齐 `x_clean`。原先仅有速度锚点和 teacher 一致性，虽然理论上能得到 clean 端点，但对 PSNR / SSIM / RMSE 这类端点指标不够直接。

2. 新增 `start_pair_prob`
   以 0.35 的概率强制采样第一个 `σ` pair，让训练更频繁覆盖 `σ≈1` 的起点段。这个位置正是 1 步推理实际调用的状态：`x=μ, σ=1`。

3. 加密 CFM 时间表
   `loss_fn_config.params.num_steps` 从 `18` 提升到 `40`，让相邻 `σ_n -> σ_{n+1}` 更细，teacher 一致性目标更平滑。

4. 下调 teacher 一致性权重
   `endpoint_loss_weight` 和 `consistency_loss_weight` 从 `1.0` 调到 `0.5`。这样真实速度锚点和 clean endpoint 监督会更主导，减少早期或滞后 EMA teacher 对画质指标的牵制。

## 新增和保留文件

代码：

```text
sgm/models/diffusion_cfm.py
sgm/modules/diffusionmodules/loss_cfm.py
sgm/modules/diffusionmodules/sampling_cfm.py
sgm/modules/diffusionmodules/sigma2st_cfm.py
sgm/modules/diffusionmodules/denoiser_scaling_cfm.py
```

配置：

```text
configs/example_training/cuhk_cfm.yaml
configs/example_training/cuhkv2_cfm.yaml
```

文档：

```text
Enhance.md
Operate.md
```

## 代码收敛

旧的独立优化入口已从代码、配置和文档中移除。原始 EMRDM 主干只保留基础扩散实现；新增优化逻辑统一放在 CFM 专属文件中。

同时从以下原始文件中移除了不再维护的旧优化类：

```text
sgm/modules/diffusionmodules/loss.py
sgm/modules/diffusionmodules/sampling.py
sgm/models/diffusion.py
```

## 配置变更摘要

`cuhk_cfm.yaml` 和 `cuhkv2_cfm.yaml` 已同步：

| 参数 | 当前值 |
|------|--------|
| `sigma_st_config.target` | `sgm.modules.diffusionmodules.sigma2st_cfm.ConsistencyFlowMatchingSigma2St` |
| `denoiser.scaling_config.target` | `sgm.modules.diffusionmodules.denoiser_scaling_cfm.ConsistencyFlowMatchingScaling` |
| `endpoint_loss_weight` | `0.5` |
| `consistency_loss_weight` | `0.5` |
| `velocity_anchor_loss_weight` | `1.0` |
| `clean_endpoint_loss_weight` | `1.0` |
| `cloud_mask_key` | `"M"` |
| `cloud_loss_weight` | `2.0` |
| `cloud_weight_velocity_anchor` | `True` |
| `start_pair_prob` | `0.35` |
| `num_steps` | `40` |
| `sampler.num_steps` | `1` |
| `loss_type` | `"charbonnier"` |
| `charbonnier_eps` | `1.0e-3` |

## 后续建议

1. 跑 CUHK-CR1 的短训练，对比优化前后的 `final_PSNR`、`final_SSIM` 和 `final_RMSE`。
2. 若 1 步指标仍偏低，优先测试 `sampler.num_steps=3`，确认多步 CFM 是否能追回细节。
3. 后续所有画质优化都建议以 ablation 配置单独验证，避免把速度收益和指标收益混在一起解释。

## 画质指标优化 — 已实装

### 5. Charbonnier loss（替换 L2）

L2 对大误差惩罚过重，导致输出偏模糊。Charbonnier `√(x²+ε²)` 是平滑 L1，对图像恢复任务普遍优于 L2。

**改动文件：**

- `sgm/modules/diffusionmodules/loss_cfm.py`：`_get_loss` 新增 `"charbonnier"` 分支；`__init__` 的 assert 同步放开，并新增可配置参数 `charbonnier_eps`。
- `configs/example_training/cuhk_cfm.yaml` 和 `cuhkv2_cfm.yaml`：`loss_type` 从 `"l2"` 改为 `"charbonnier"`，`charbonnier_eps` 默认设为 `1.0e-3`。

**核心公式：**

$$\mathcal{L}_{charb}=\frac{1}{N}\sum\sqrt{(pred-target)^2+\varepsilon^2},\quad\varepsilon=10^{-3}$$

**预期效果：** PSNR / SSIM 有机会小幅提升，输出细节更清晰；ε 越小越接近 L1，ε 越大则在小误差区域更平滑。

### 6. 云区加权 loss

CFM loss 新增 `cloud_mask_key`、`cloud_loss_weight`、`cloud_weight_velocity_anchor`。当前 CUHK / CUHKV2 CFM 配置使用 `batch["M"]` 作为云区 mask，`cloud_loss_weight=2.0`，对 clean endpoint 监督和速度锚点监督进行云区加权。

权重图会按样本均值归一化：

$$w=\frac{1+(\lambda_{cloud}-1)M}{mean(1+(\lambda_{cloud}-1)M)}$$

这样云区像素梯度更强，但单个样本的平均 loss 尺度保持接近原始设置。

---

## 可选画质性能优化方向

以下方向中，Charbonnier、CUHK 成对数据增强、云区加权 loss 和 TTA 推理已实装；其余适合作为下一轮 CFM 画质指标优化的实验候选。推荐后续优先从 `SSIM endpoint loss` 和 endpoint fine-tuning 开始，它们最直接服务 PSNR / SSIM / RMSE 这类端点指标。

| 方向 | 主要目标 | 实现思路 | 风险 / 注意点 |
|------|----------|----------|---------------|
| SSIM / MS-SSIM endpoint loss | 提升 SSIM 和视觉结构 | 在 `clean_endpoint_loss` 旁加入 `1 - SSIM(f_student, x_clean)`，权重可从 `0.05~0.2` 起试 | 权重大时可能轻微牺牲 PSNR，需要单独扫权重 |
| 云区加权 loss ✅ | 强化真正被云遮挡区域的恢复 | 已在 `loss_cfm.py` 中使用 `batch["M"]`；默认对 `clean_endpoint_loss` 加权，并可对 velocity anchor 同步加权；`cloud_loss_weight=2.0` | mask 质量差会误导训练；建议继续扫 `1.5/2.0/3.0` |
| endpoint fine-tuning 阶段 | 进一步优化 1 步输出端点 | 训练后期降低学习率，设置更高 `clean_endpoint_loss_weight` 和 `start_pair_prob`，同时降低 teacher consistency 权重 | 过度 fine-tune 可能降低多步采样稳定性，建议只针对最终 1 步模型使用 |
| 2-3 步 CFM 评估 | 以较小速度代价换画质 | 推理时把 `sampler.num_steps` 从 `1` 提到 `2` 或 `3`，使用 CFM 内置 Euler 多步路径 | 速度随步数线性增加；主表需明确标注步数 |
| 梯度 / 边缘 loss | 改善建筑、道路和地物边界 | 对 `f_student` 和 `x_clean` 加 Sobel / finite-difference loss，权重可从 `0.05~0.1` 起试 | 可能放大噪声或边缘伪影，不建议权重过大 |
| 光谱一致性 loss | 改善多光谱真实性 | 在多波段输出上加入 per-band loss、band ratio consistency 或 SAM loss | SAM / ratio loss 对接近 0 的波段值敏感，需要 clamp 或 epsilon |
| hard example / hard crop sampling | 提升厚云和复杂地表样本表现 | 提高厚云、云边界、纹理密集 crop 的采样概率 | 需要统计样本难度；采样过偏会影响整体分布 |
| EMA decay 调度 | 改善 teacher 跟随速度 | 前期使用较低 decay（如 `0.999`），后期升到 `0.9999`；或固定试 `0.9995` | teacher 太快会不稳定，太慢会滞后；需要和 warmup 一起调 |
| Charbonnier loss ✅ | 减少 L2 导致的模糊 | 已实装，见上方"已实装"章节 | 改动最小；ε 越小越接近 L1，默认 `1.0e-3` |
| LPIPS 感知 loss | 改善视觉结构和 SSIM | 环境已有 `lpips==0.1.4`，在 `forward` 里对 `f_student` 和 `x_clean` 加 `lpips.LPIPS(net='alex')` 项，权重从 `0.01~0.05` 起试 | 需要 3 通道输入，多波段需先选 RGB 子集；权重过大会牺牲 PSNR |
| DCT 频域 loss | 补偿高频细节和纹理 | 环境已有 `dctorch==0.1.2`，对 `f_student` 和 `x_clean` 做 `dct_2d` 后算 L2；可对高频系数额外加权 | 高频权重过大会放大噪声；建议先用均匀权重验证方向 |
| 多尺度 endpoint loss | 稳定全局色彩同时保细节 | 对 `f_student` 和 `x_clean` 分别在 1×、0.5×、0.25× 分辨率下算 `clean_endpoint_loss`，低分辨率权重可略高 | 实现稍复杂；下采样方式（bilinear/area）影响结果，建议用 area |
| Test-Time Augmentation（TTA） ✅ | 推理阶段零成本提升 PSNR | 已在 `sampling_cfm.py` 中实装，4 种几何变换（原图 + 水平翻转 + 垂直翻转 + 180° 旋转）推理后平均输出；YAML 中设置 `tta: True` 或 CLI 追加参数即可启用 | PSNR 通常涨 0.1~0.3 dB；推理时间变为 4×，需在论文中注明 |
| 自适应 σ 采样 | 让模型更多训练难学的 σ 段 | 训练中统计各 σ 区间的平均 loss，按 loss 大小动态调整采样概率，替代固定的 `start_pair_prob` | 需要额外统计开销；采样过偏可能破坏整体分布，建议设置采样概率下界 |

建议实验顺序：

1. `Charbonnier loss`：改动最小，先验证是否稳定提升 PSNR/SSIM，再叠加其他项。
2. `LPIPS loss`：库已装，小权重叠加，主要改善 SSIM 和视觉结构。
3. `SSIM endpoint loss`：先看 SSIM 是否稳定提升，以及 PSNR 是否可接受。
4. `DCT 频域 loss`：库已装，补高频细节，与 Charbonnier 叠加效果互补。
5. `TTA 推理`：已实装，零训练成本，可在任意阶段直接测试。
6. 云区加权 loss：已实装，建议扫 `cloud_loss_weight=1.5/2.0/3.0` 并观察云区 / 非云区指标是否同时稳定。
7. endpoint fine-tuning：作为训练后期小学习率阶段，而不是从头训练就强行提高 endpoint 权重。
8. `sampler.num_steps=2/3`：作为质量/速度折中结果加入对比表。
9. 多尺度 loss、自适应 σ 采样：收益不确定，作为后期消融实验候选。

## 可调参数建议

按影响优先级排列，适合在训练或评估时尝试调整：

### 损失权重（最直接影响画质指标）

| 参数路径 | 当前值 | 调整建议 |
|----------|--------|----------|
| `loss_fn_config.params.clean_endpoint_loss_weight` | `1.0` | 提高至 `1.5~2.0`，加强 PSNR/SSIM 直接监督 |
| `loss_fn_config.params.velocity_anchor_loss_weight` | `1.0` | 适当提高，使速度场更准确 |
| `loss_fn_config.params.endpoint_loss_weight` | `0.5` | 降低可减少 teacher 噪声干扰 |
| `loss_fn_config.params.consistency_loss_weight` | `0.5` | 同上，与 endpoint_loss_weight 联动调整 |
| `loss_fn_config.params.cloud_loss_weight` | `2.0` | 已启用云区加权，可扫 `1.5/2.0/3.0`；过高可能损伤非云区色彩 |
| `loss_fn_config.params.start_pair_prob` | `0.35` | 提高至 `0.5` 可增强 1 步推理能力，但可能损失多步泛化 |

### 推理步数（测试时调，无需重训）

从 1 改到 3 通常有明显画质提升，代价是推理耗时增加约 3 倍：

```bash
model.params.sampler_config.params.num_steps=3
```

### 学习率调度

| 参数路径 | 当前值 | 调整建议 |
|----------|--------|----------|
| `model.base_learning_rate` | `1e-4` | 可试 `5e-5`（更稳定）或 `2e-4`（更快收敛） |
| `scheduler_config.params.max_decay_steps` | `400000` | 若训练步数超过此值应相应增大 |
| `loss_fn_config.params.consistency_warmup_steps` | `2000` | 训练不稳定时可增大至 `5000` |

### 其他

| 参数路径 | 当前值 | 调整建议 |
|----------|--------|----------|
| `model.params.teacher_ema_decay` | `0.9999` | 训练初期可降至 `0.999`，让 teacher 更新更快 |
| `loss_fn_config.params.num_steps` | `40` | 增大至 `80~100` 可降低 σ pair 误差，但训练变慢 |
| `data.params.batch_size` | `2` | 显存允许时增大，训练更稳定 |

## 代码层面提升 PSNR/SSIM 的方向

以下是通过代码审查发现的、有潜力提升画质指标的方向，包含已实装项和后续实验候选，按预期收益排序：

### 1. 数据增强（翻转/90度旋转，已实装）

已在 `sgm/data/cuhk/image_datasets.py` 中为 CUHK CFM 训练加入基础成对数据增强。`TrainDataset` 新增 `augment`、`hflip_p`、`vflip_p`、`rot90_p` 参数；`_augment_pair()` 会对 clean label `t` 和 cloudy input `x` 同步执行水平翻转、垂直翻转和 90 度整数倍旋转。

增强放在 `imresize()` 之后、`M` mask 计算之前，因此 `label`、`cond_image` 和 `M` 的空间位置保持一致。当前只在 `cuhk_cfm.yaml` 和 `cuhkv2_cfm.yaml` 的 `train` dataloader 中开启，validation / test / predict 不开启增强。

预期提升：+0.2~0.5 dB PSNR。

```yaml
train:
  target: sgm.data.cuhk.image_datasets.TrainDataset
  params:
    augment: True
    hflip_p: 0.5
    vflip_p: 0.5
    rot90_p: 0.5
```

### 2. SSIM endpoint loss

代码库中 `sgm/modules/learning/pytorch_ssim/` 已有可微分 SSIM 实现，但训练时未使用。在 `loss_cfm.py` 中新增第 5 个损失项：

```python
ssim_loss = 1 - ssim(f_student, x_clean)
```

建议权重从 `0.05~0.2` 起试，直接优化 SSIM 指标。

### 3. LPIPS 感知 loss

`lpips==0.1.4` 已安装且评估时已使用，但训练时未参与。小权重（`0.01~0.05`）叠加可改善结构保真度和 SSIM。注意 LPIPS 只支持 3 通道输入，需要先取 RGB 子集。

### 4. 云区加权 loss

已在 CFM 专属损失 `sgm/modules/diffusionmodules/loss_cfm.py` 中实装。数据加载时已有云 mask `M = np.clip((t-x).sum(axis=2), 0, 1)`，loss 现在直接从 batch 读取 `cloud_mask_key: "M"`，不需要把 mask 通过 `batch2model_keys` 传给网络。

实现方式：
- `_get_cloud_weight()` 将 `[B,H,W]` 或 `[B,1,H,W]` 的 `M` 转为像素权重 `1 + (cloud_loss_weight - 1) * M`。
- 权重按每个样本的均值归一化，避免云比例不同导致整体 loss 尺度漂移。
- 默认加权 `clean_endpoint_loss`；当 `cloud_weight_velocity_anchor: True` 时，同步加权 `velocity_anchor_loss`。
- `cuhk_cfm.yaml` 和 `cuhkv2_cfm.yaml` 当前默认 `cloud_loss_weight: 2.0`，建议后续做 `1.5/2.0/3.0` 消融。

### 5. Test-Time Augmentation（TTA，已实装）

零训练成本。推理时对输入做 4 种几何变换（原图、水平翻转、垂直翻转、水平+垂直翻转/180° 旋转），分别推理后逆变换并取平均输出。预期提升 +0.1~0.3 dB PSNR，推理时间变为 4 倍。

**改动文件：**

- `sgm/modules/diffusionmodules/sampling_cfm.py`：新增 `tta` 构造参数、`_apply_transform_to_cond()` / `_sample_single()` / `_sample_with_tta()` 方法。在 `__call__` 入口处，当 `self.tta=True` 时走 TTA 路径。TTA 对 `mu` 和所有 4-D cond 张量（如拼接的 cloudy image）同步变换，保证空间一致性。
- `configs/example_training/cuhk_cfm.yaml` 和 `cuhkv2_cfm.yaml`：`sampler_config.params` 新增 `tta: False`，默认关闭。

**启用方式：**

方法 1 — 修改 YAML：
```yaml
sampler_config:
  params:
    tta: True
```

方法 2 — CLI 覆盖（无需修改文件）：
```bash
python main.py --base configs/example_training/cuhk_cfm.yaml \
    -t false \
    model.params.sampler_config.params.tta=True
```

**注意事项：**
- TTA 模式下不返回 `intermediates` / `denoiseds`（多变换平均后无意义），image_logger 的中间步可视化会变为空。建议仅在 test / predict 时开启，训练时的 validation 保持 `tta: False`。
- TTA 与 `num_steps>1` 多步采样兼容，可叠加使用（推理时间为 4×num_steps）。

### 6. 输出 clamp

当前 CFM sampler 输出 `x_clean` 后没有做值域裁剪，网络输出可能超出 [-1, 1] 范围导致异常像素。在 `sampling_cfm.py` 的输出处添加 `x.clamp(-1, 1)` 是最简单的兜底。

### 7. endpoint fine-tuning 阶段

训练后期从已有 checkpoint 继续训练，降低学习率（如 `base_learning_rate: 2e-5`），同时提高 `clean_endpoint_loss_weight` 到 `2.0~3.0`、提高 `start_pair_prob` 到 `0.5~0.6`、降低 `endpoint_loss_weight` 和 `consistency_loss_weight` 到 `0.2`，让训练完全聚焦于 1 步输出的画质端点。

### 8. EMA teacher decay 调度

当前 `teacher_ema_decay` 固定为 `0.9999`。可改为前 5000 步使用 `0.999`（teacher 快速跟随），之后线性升至 `0.9999`（teacher 稳定）。需要修改 `diffusion_cfm.py` 中的 `_update_teacher()` 方法。

### 9. 梯度/边缘 loss

对 `f_student` 和 `x_clean` 施加 Sobel 或 finite-difference 算子后计算 loss，权重 `0.05~0.1`，改善建筑、道路等地物边界的恢复效果。

### 建议实验顺序

| 优先级 | 方向 | 是否需要重新训练 | 预期 PSNR 提升 |
|--------|------|-----------------|---------------|
| 1 | 数据增强（翻转/90度旋转） | 是 | 已实装，建议先短训验证 +0.2~0.5 dB |
| 2 | TTA 推理 | 否 | 已实装，建议在已有 checkpoint 上直接测试 +0.1~0.3 dB |
| 3 | SSIM endpoint loss | 是 | +0.1~0.3 dB |
| 4 | 云区加权 loss | 是 | 已实装，建议扫 `1.5/2.0/3.0` |
| 5 | 输出 clamp | 否 | 防止异常值拉低均值 |
| 6 | LPIPS loss | 是 | 主要改善 SSIM |
| 7 | endpoint fine-tuning | 是（微调） | +0.1~0.2 dB |
| 8 | EMA decay 调度 | 是 | 不确定 |
| 9 | 梯度/边缘 loss | 是 | +0.05~0.1 dB |



本次通读代码后，除前面已经列出的画质优化外，还有几类更偏工程和评估可信度的提升方向。它们不一定直接改变网络结构，但会影响实验能否稳定复现、预测流程能否跑通，以及 PSNR / SSIM / RMSE 是否能真实反映去云质量。

| 优先级 | 方向 | 触发代码 / 现象 | 建议处理 |
|--------|------|----------------|----------|
| P0 | 预测 batch size 与 `predict_step` 对齐 | `ResidualDiffusionEngine.predict_step()` 和 `TemporalResidualDiffusionEngine.predict_step()` 都断言 `batch size == 1`，但 `cuhk_cfm.yaml` / `cuhkv2_cfm.yaml` 当前 `data.params.batch_size=4` | 为 `DataModuleFromConfig` 增加 `predict_batch_size`，或在预测命令 / 配置里单独强制 `batch_size=1`；否则 `--predict` 会直接触发断言 |
| P0 | 评估数值稳定性 | `metrics.py` 中 PSNR 使用 `20 * log10(1 / rmse)`，SAM 分母也没有 epsilon；完美预测或全零波段可能产生 `inf` / `nan` | 在 RMSE 和光谱范数分母加入 `eps`，并在 `avg_img_metrics.add()` 中明确处理 `inf`；这样结果表不会被异常值污染 |
| P1 | 云区 / 非云区分区指标接入主评估 | `metrics.img_metrics()` 已支持 `masks`，CUHK dataloader 也返回 `M`，但 `shared_test_step()` / `predict_step()` 调用指标时没有传 mask | 把 `M` 或 `masks` 通过评估链路传入，输出 `RMSE_cloudy`、`RMSE_cloudfree` 等指标；这比单一全图 RMSE 更能说明去云区域是否真的改善 |
| P1 | 推理输出统一 clamp / range policy | `sampling_cfm.py` 返回 `x_clean` 后没有统一裁剪，后续评估和保存才做部分 `scale_01` | 在 sampler 或 decode 后建立统一策略：默认 `clamp(-1, 1)`，同时保留可关闭开关，避免异常像素影响指标和 GeoTIFF 保存 |
| P1 | 成对数据增强扩展与复现性 | CUHK CFM 已实装水平翻转、垂直翻转和 90 度旋转；Sen2_MTC_New 的增强仍依赖可变的 `self.index`，在 shuffle / 多 worker 下不够可复现 | 先对 CUHK 增强做短训消融；后续将同样的配置式同步增强扩展到多时相数据，并让随机数按样本 index 或 worker seed 生成 |
| P1 | 数据缓存路径可配置 | Sentinel dataloader 会在数据目录写 `path.txt` 和 `_mask.npy`，只读数据盘或多进程首次运行时可能出错或竞争 | 增加独立 `cache_dir`，缓存文件按 split / cloud mask 类型命名；写缓存时使用临时文件再原子替换 |
| P2 | DataModule 支持分阶段 batch size | 当前 train / val / test / predict 共用同一个 `batch_size`，但训练、测试、预测的显存与保存需求不同 | 支持 `train_batch_size`、`val_batch_size`、`test_batch_size`、`predict_batch_size`，默认回退到原 `batch_size`，减少为了预测而牺牲训练吞吐 |
| P2 | RGB-only CUHK 兼容性修复 | `TrainDataset.__init__()` 默认允许 `nir_datasets_dir=None`，但随后直接 `len(self.nir_imlistl)` 会在无 NIR 时出错 | 仅在 `nir_datasets_dir is not None` 时检查 RGB / NIR 数量一致；同时根据通道数自动调整 `network_config.in_channels/out_channels` |
| P2 | 轻量 smoke tests / 配置检查 | 当前 `test_k_diffusion.py` 更像 FLOPs 脚本，依赖 CUDA 和 `calflops`，不适合作为快速回归测试 | 增加 CPU 可跑的 smoke tests：CFM loss 前向、1 步 sampler shape/range、DataModule 小样本切分、metrics 零误差稳定性、YAML target 可实例化检查 |

### 建议加入下一轮实验 / 开发顺序

1. 先修 `predict_batch_size=1` 和评估 `eps`，这是避免流程失败和指标异常的基础项。
2. 接着接入 mask 分区指标与统一输出 clamp，用于更可靠地观察后续画质优化收益。
3. 然后对已实装的 CUHK 成对数据增强做短训消融，确认收益后再扩展到 Sen2_MTC_New / Sentinel 多时相场景。
4. 最后补 smoke tests 和可配置缓存目录，降低后续继续叠加 CFM loss 等优化时的回归成本。

## V1.3版本说明
1. 加入数据增强（水平垂直翻转/90度旋转）相关优化。
   - `sgm/data/cuhk/image_datasets.py` 新增 `augment`、`hflip_p`、`vflip_p`、`rot90_p` 配置项。
   - `_augment_pair()` 对 `label` 和 `cloud` 同步执行水平翻转、垂直翻转和 90 度整数倍旋转。
   - 增强发生在 resize 之后、`M` mask 计算之前，避免监督图、条件图和 mask 错位。
   - `configs/example_training/cuhk_cfm.yaml` 与 `cuhkv2_cfm.yaml` 仅在 `train` 数据集开启增强，验证、测试和预测保持确定性输入。
2. 加入云区加权 loss。
   - `sgm/modules/diffusionmodules/loss_cfm.py` 新增 `cloud_mask_key`、`cloud_loss_weight`、`cloud_weight_velocity_anchor` 配置项。
   - 默认从 batch 的 `M` 读取云区 mask，支持 `[B,H,W]` 和 `[B,1,H,W]`，尺寸不一致时用 nearest 插值对齐到训练图大小。
   - 权重公式为 `1 + (cloud_loss_weight - 1) * M`，并按样本均值归一化，避免云比例改变整体 loss 尺度。
   - 当前对 `clean_endpoint_loss` 启用云区加权，并通过 `cloud_weight_velocity_anchor: True` 同步加权速度锚点；teacher endpoint / velocity consistency 仍保持原权重，避免放大早期 teacher 噪声。
   - `configs/example_training/cuhk_cfm.yaml` 与 `cuhkv2_cfm.yaml` 默认启用 `cloud_loss_weight: 2.0`，后续建议短训扫描 `1.5/2.0/3.0`。
3. 加入 Test-Time Augmentation（TTA）推理。
   - `sgm/modules/diffusionmodules/sampling_cfm.py` 新增 `tta` 构造参数（默认 `False`），及 `_apply_transform_to_cond()`、`_sample_single()`、`_sample_with_tta()` 三个方法。
   - 启用 TTA 时，采样器对输入执行 4 种几何变换（原图、水平翻转、垂直翻转、水平+垂直翻转），分别推理后逆变换并取平均。对 `mu` 和所有 4-D cond 张量同步变换，保证空间一致性。
   - `configs/example_training/cuhk_cfm.yaml` 和 `cuhkv2_cfm.yaml` 的 `sampler_config.params` 新增 `tta: False`，可通过 YAML 修改或 CLI 覆盖 `model.params.sampler_config.params.tta=True` 启用。
   - 预期提升 +0.1~0.3 dB PSNR，推理时间变为 4 倍，不需要重新训练。
   - TTA 模式不返回 `intermediates` / `denoiseds`，建议仅在 test / predict 阶段开启。
