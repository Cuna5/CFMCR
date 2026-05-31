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

## 下一阶段画质提升路线

前几轮改动说明，继续堆采样技巧和零散 loss 候选收益很有限。当前 CFM 更像一步端点预测器，核心优化应回到训练目标和数据分布本身。后续建议只保留下面几条主线，其他 LPIPS、DCT、边缘 loss、EMA 调度等方向暂时降级为备选，不再作为近期重点。

### 1. 回到确定性 1-step CFM

`s_churn` 对当前 CFM 不适合作为画质指标优化主线。CFM 训练点在确定性 OT 直线路径上，没有学习额外高斯噪声去噪；加入扰动会把状态推离训练分布，容易导致 PSNR / SSIM / RMSE 明显下降，严重时输出噪声图。

建议先固定如下基线：

```yaml
sampler_config:
  params:
    num_steps: 1
    tta: False
    s_churn: 0.0
```

在这个基线稳定后，再单独评估 `tta=True`。多步和扰动不再作为默认提分手段。

### 2. 加 MS-SSIM endpoint loss ✅

已实装。`loss_cfm.py` 新增 `_ms_ssim_loss` 方法（纯 PyTorch，无额外依赖），在 `_forward` 中当 `ssim_endpoint_loss_weight > 0` 时叠加到 `clean_endpoint_loss` 上：

```text
total_loss += ssim_endpoint_loss_weight * (1 - MS_SSIM(f_student, x_clean))
```

两个 YAML 均已加入参数（默认 `0.0`，不影响现有训练）：

```yaml
loss_fn_config:
  params:
    ssim_endpoint_loss_weight: 0.0   # scan 0.02/0.05/0.1/0.2
```

注意事项：

- MS-SSIM 输入建议转到 `[0, 1]`，即 `(x.clamp(-1, 1) + 1) / 2`。
- 当前 CUHK CFM 输出为 4 通道，MS-SSIM 可优先只对 RGB 三通道计算，NIR 继续由 Charbonnier / RMSE 约束。
- 权重过大可能提升 SSIM 但损伤 PSNR / RMSE，需要单独消融。

### 3. 做 endpoint fine-tuning

不要一开始就把所有权重调得很激进。更稳的方式是先用当前配置训练出稳定 checkpoint，再进入小学习率微调阶段，把训练目标集中到一步端点画质。

建议 fine-tuning 起点：

```yaml
model:
  base_learning_rate: 2.0e-5
  params:
    ckpt_path: "/path/to/stable_cfm.ckpt"
    loss_fn_config:
      params:
        clean_endpoint_loss_weight: 2.0
        velocity_anchor_loss_weight: 1.0
        endpoint_loss_weight: 0.2
        consistency_loss_weight: 0.2
        start_pair_prob: 0.5
```

推荐扫描：

| 参数 | 参考范围 |
|------|----------|
| `base_learning_rate` | `1.0e-5 / 2.0e-5 / 5.0e-5` |
| `clean_endpoint_loss_weight` | `1.5 / 2.0 / 3.0` |
| `start_pair_prob` | `0.35 / 0.5 / 0.6` |
| `endpoint_loss_weight` / `consistency_loss_weight` | `0.1 / 0.2 / 0.5` |

该阶段只服务最终 1-step 指标，不再追求多步泛化。

### 4. 加 hard / cloud crop sampling

如果全图大部分区域本来就干净，模型容易学成保守复原，指标提升也会很慢。更有效的方向是提高训练中厚云、云边界、纹理复杂区域的出现概率。

建议做法：

- 根据 `M` mask 统计每张图或每个 crop 的云量比例。
- 提高厚云区域、云边界区域、纹理密集区域 crop 的采样概率。
- 保留一部分普通样本，避免采样过偏导致整体色彩分布漂移。

推荐起步比例：

| 样本类型 | 采样占比 |
|----------|----------|
| 普通随机 crop | `40%~60%` |
| 高云量 crop | `20%~40%` |
| 云边界 / 复杂纹理 crop | `10%~30%` |

这类数据分布改动通常比继续微调小 loss 更可能带来可见收益。

### 5. 大模型伪无云图辅助监督

可以离线使用大模型为训练集 cloudy image 生成伪无云图，将这些图作为辅助监督引导 CFM 恢复云区结构。该方向可能改善云区纹理和 SSIM，但对 PSNR / RMSE 有风险：大模型生成图如果和真实 clean label 像素、颜色或光谱不完全一致，可能视觉更好但 PSNR 下降。

核心原则：

- 伪无云图只作为辅助监督，不替代真实 `label`。
- 优先只在云区使用 pseudo loss，避免改坏原本干净区域。
- 权重要小，建议从 `0.02~0.05` 开始。
- 如果大模型只生成 RGB，则只监督 RGB 三通道；NIR 仍由真实 label 监督。
- 推理阶段不依赖大模型，避免训练 / 推理条件不一致。

#### 数据准备

推荐保存为和原训练集同名的目录结构：

```text
CUHK-CR2_pseudo/
  train/
    label/
      xxx.png
      yyy.png
```

在 dataloader 中新增配置：

```yaml
train:
  params:
    pseudo_clean_dir: "/path/to/CUHK-CR2_pseudo"
    pseudo_rgb_only: True
```

实现时需要在 `sgm/data/cuhk/image_datasets.py` 中读取同名 pseudo 图，并和 `label / cond_image` 同步 resize、同步数据增强，最后返回：

```python
{
    "pseudo_clean": pseudo,
}
```

同步增强很重要，否则 pseudo 图和真实 label / cloudy input 会空间错位。

#### Loss 接入

在 `sgm/modules/diffusionmodules/loss_cfm.py` 中，基于当前 endpoint 输出：

```python
f_student = x_tn + sigma_n_bc * v_student
```

增加低权重 pseudo endpoint loss：

```text
L_pseudo = Charbonnier(f_student_rgb, pseudo_clean_rgb)
L_total += pseudo_clean_loss_weight * L_pseudo
```

推荐新增参数：

```yaml
loss_fn_config:
  params:
    pseudo_clean_key: "pseudo_clean"
    pseudo_clean_loss_weight: 0.02
    pseudo_clean_rgb_only: True
    pseudo_clean_cloud_only: True
    pseudo_clean_start_step: 2000
```

其中 `pseudo_clean_start_step` 用于前期先让模型学习真实 clean label，再逐步引入伪标签辅助，降低被生成图带偏的风险。

#### 云区和置信度加权

最简单的方式是只在 `M` 云区监督：

```text
pseudo_weight = M
```

如果训练集有真实 clean label，可以进一步用 pseudo 和真实 label 的差异构造置信度：

```text
confidence = exp(-abs(pseudo_clean - label) / tau)
pseudo_weight = M * confidence
```

这样 pseudo 与真实 label 差距过大的区域会自动降权，更有利于 PSNR / RMSE。

#### 是否值得做的筛选标准

在正式训练前，先统计训练集上：

```text
PSNR(pseudo_clean, label)
PSNR(cloudy, label)
SSIM(pseudo_clean, label)
SSIM(cloudy, label)
```

如果 `PSNR(pseudo_clean, label)` 至少比 `PSNR(cloudy, label)` 高 `1~2 dB`，低权重云区辅助更值得尝试。若 pseudo 本身 PSNR 不高，则它更可能只改善视觉或 SSIM，甚至损伤 PSNR。

推荐微调策略：

```yaml
model:
  base_learning_rate: 2.0e-5
  params:
    ckpt_path: "/path/to/stable_cfm.ckpt"
    loss_fn_config:
      params:
        clean_endpoint_loss_weight: 2.0
        pseudo_clean_loss_weight: 0.02
        pseudo_clean_rgb_only: True
        pseudo_clean_cloud_only: True
        endpoint_loss_weight: 0.2
        consistency_loss_weight: 0.2
        start_pair_prob: 0.5
```

建议消融：

| 实验 | 设置 |
|------|------|
| baseline | 不使用 pseudo clean |
| pseudo low | `pseudo_clean_loss_weight=0.02` |
| pseudo mid | `pseudo_clean_loss_weight=0.05` |
| pseudo cloud-only | 只在 `M` 云区启用 |
| pseudo confidence | 云区 + pseudo/label 置信度加权 |

### 6. 输出 clamp 作为指标兜底

CFM sampler 输出如果超出 `[-1, 1]`，少量异常像素也可能拖低 RMSE / PSNR。建议在 sampler 最终输出处加可配置 clamp：

```python
x_clean = x_clean.clamp(-1, 1)
```

建议新增参数：

```yaml
sampler_config:
  params:
    clamp_output: True
```

这不是核心提分策略，但可以避免异常值污染指标和保存结果。

### 7. 如果要多步，就训练多步

当前多步推理只是把一步 CFM 速度场拿来做 Euler 积分。模型训练时看到的是理想 OT 路径点；推理多步时前一步误差会让下一步状态偏离训练分布，因此步数增加后指标下降是合理现象。

如果目标是让 `num_steps=2/3/4` 真正优于 1 步，需要加入 sampler-aware training：

1. 从 `x=mu, sigma=sigma_max` 开始。
2. 用当前 sampler 更新一步得到 `x_next`。
3. 把 `x_next` 再喂回模型，展开 2-4 步。
4. 对最终 endpoint 计算 `x_clean` loss。

这会增加训练显存和时间，但它和“只在推理时调大步数”不是同一件事。近期若主目标是指标，优先保持 1-step。

### 当前推荐实验顺序

1. 固定确定性 1-step：`num_steps=1, s_churn=0.0, tta=False`，得到稳定基线。
2. 加 MS-SSIM endpoint loss，扫 `ssim_endpoint_loss_weight=0.02/0.05/0.1/0.2`。
3. 基于稳定 checkpoint 做 endpoint fine-tuning。
4. 加 hard / cloud crop sampling，重新训练或微调。
5. 统计 pseudo clean 质量；若 pseudo 明显优于 cloudy，再做低权重云区辅助微调。
6. 加输出 clamp，确认异常值不会拖低指标。
7. 最后单独评估 `tta=True`，作为无需重训的推理增强。

### 暂不作为近期主线的方向

- `s_churn`：当前已验证可能导致噪声图，除非重新设计带噪训练，否则不建议继续投入。
- 单纯增加 `sampler.num_steps`：没有多步训练配合时，指标可能下降。
- LPIPS / DCT / 边缘 loss：可能改善视觉观感，但对 PSNR / RMSE 不一定友好，先不作为主线。
- EMA decay 调度：收益不确定，优先级低于 endpoint loss 和数据采样。

## V2.0版本说明

V2.0 同步 `NIR_ME_CFM方案V3.md` 中的创新点 1：`NCB`
（NIR-guided Conditional Backbone）。本次只落地 NCB，不改 CFM 的
loss / sampler / denoiser / Engine，也不改变主干输入的 `in_channels=8`。

### 1. NCB 数据条件

从推理阶段可获得的含云 RGBNIR 条件图 `cond_image` 派生辅助条件：

```text
cond_image: [B, 4, H, W], range [-1, 1]
aux_cond:   [B, 5, H, W]
```

`aux_cond` 的 5 个通道为：

```text
NDVI_cloudy + Edge_R + Edge_G + Edge_B + Edge_NIR
```

实现位置：

```text
sgm/data/cuhk/image_datasets.py
```

关键约束：

- `aux_cond` 只由含云 `cond_image` 构造，不使用真实 label。
- 当前 dataloader 中的 `M` 仍只用于 loss weighting，不进入 NCB，避免真值泄漏。
- `aux_cond` 在数据增强之后、归一化到 `[-1, 1]` 之后生成，保证和 `cond_image / label` 空间对齐。

### 2. 条件注入方式

新增 `DictEmbedder`，让 `aux_cond` 作为独立 key 进入网络，而不是拼到
`concat` 条件里：

```text
sgm/modules/encoders/modules.py
```

配置中新增：

```yaml
- is_trainable: False
  input_key: "aux_cond"
  ucg_rate: 0.0
  target: sgm.modules.encoders.modules.DictEmbedder
  params:
    output_key: "aux_cond"
```

这样主干输入仍保持：

```text
cat(x_t, cond_image) -> [B, 8, H, W]
```

`CloudRemovalWrapper` 已经会把非 `"concat"` 的条件继续透传给网络，因此本次无需修改 wrapper。

### 3. NIR-guided Conditional Backbone

在 Hourglass Transformer 中新增轻量全局辅助编码器：

```text
sgm/modules/diffusionmodules/k_diffusion/image_transformer.py
```

数据流：

```text
aux_cond -> AuxGlobalEncoder -> aux_global
time_emb -> MappingNetwork -> cond
cond' = cond + aux_to_cond(aux_global)
```

其中 `aux_to_cond` 使用 zero-init：

```text
init(aux_to_cond.weight) = 0
```

因此从旧 CFM checkpoint 热启动时，初始状态下：

```text
cond' = cond
```

新增模块不会破坏原模型行为，训练开始后再逐步学习 NDVI / Edge 对 backbone 条件调制的贡献。

### 4. 配置同步

以下两个 CFM 配置已打开 NCB only：

```text
configs/example_training/cuhk_cfm.yaml
configs/example_training/cuhkv2_cfm.yaml
```

新增网络参数：

```yaml
network_config:
  params:
    use_aux_cond: true
    aux_channels: 5
    aux_hidden_channels: 64
    aux_global_mode: "nce"
```

同时增加：

```yaml
log_keys: ["cond_image"]
```

原因是 `aux_cond` 为 5 通道，不适合走默认图片日志保存流程；日志仍只记录原有含云条件图。

### 5. 当前验证状态

已完成轻量验证：

```text
python -m compileall
YAML 解析和关键字段检查
aux_cond helper 输出形状检查: (5, H, W)
```

完整模型前向在当前本机 Python 环境中未完成，原因是环境缺少
`pkg_resources / einops` 等依赖；这属于本地依赖问题，不是本次代码语法问题。

### 6. 后续训练建议

V2.0 建议先只跑 NCB 消融，不同时打开 TMM / MEF：

```text
Baseline CFM -> +NCB
```

推荐从稳定 CFM checkpoint 微调，优先观察：

```text
RMSE / PSNR / SSIM
云区边缘、农田边界、水体岸线、建筑轮廓
```

若 NCB 指标和视觉趋势稳定，再继续接入 V3 的创新点 2：TMM。
