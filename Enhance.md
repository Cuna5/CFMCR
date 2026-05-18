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

---

## 可选画质性能优化方向

以下方向尚未实装，适合作为下一轮 CFM 画质指标优化的实验候选。推荐优先从 `SSIM endpoint loss`、云区加权 loss 和 endpoint fine-tuning 开始，它们最直接服务 PSNR / SSIM / RMSE 这类端点指标。

| 方向 | 主要目标 | 实现思路 | 风险 / 注意点 |
|------|----------|----------|---------------|
| SSIM / MS-SSIM endpoint loss | 提升 SSIM 和视觉结构 | 在 `clean_endpoint_loss` 旁加入 `1 - SSIM(f_student, x_clean)`，权重可从 `0.05~0.2` 起试 | 权重大时可能轻微牺牲 PSNR，需要单独扫权重 |
| 云区加权 loss | 强化真正被云遮挡区域的恢复 | 若 batch 有 cloud mask，则对云区的 endpoint / velocity anchor loss 乘更高权重；没有 mask 时可用 `|label-cond_image|` 近似 hard 区域 | mask 质量差会误导训练；权重过高可能损伤无云区域色彩一致性 |
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
| Test-Time Augmentation（TTA） | 推理阶段零成本提升 PSNR | 推理时对输入做水平翻转、垂直翻转各预测一次，三次结果平均后输出；不改模型权重 | PSNR 通常涨 0.1~0.3 dB；推理时间变为 3×，需在论文中注明 |
| 自适应 σ 采样 | 让模型更多训练难学的 σ 段 | 训练中统计各 σ 区间的平均 loss，按 loss 大小动态调整采样概率，替代固定的 `start_pair_prob` | 需要额外统计开销；采样过偏可能破坏整体分布，建议设置采样概率下界 |

建议实验顺序：

1. `Charbonnier loss`：改动最小，先验证是否稳定提升 PSNR/SSIM，再叠加其他项。
2. `LPIPS loss`：库已装，小权重叠加，主要改善 SSIM 和视觉结构。
3. `SSIM endpoint loss`：先看 SSIM 是否稳定提升，以及 PSNR 是否可接受。
4. `DCT 频域 loss`：库已装，补高频细节，与 Charbonnier 叠加效果互补。
5. `TTA 推理`：零训练成本，可在任意阶段直接测试。
6. 云区加权 loss：优先在有可靠 cloud mask 的数据集上做。
7. endpoint fine-tuning：作为训练后期小学习率阶段，而不是从头训练就强行提高 endpoint 权重。
8. `sampler.num_steps=2/3`：作为质量/速度折中结果加入对比表。
9. 多尺度 loss、自适应 σ 采样：收益不确定，作为后期消融实验候选。

## V1.3版本说明
1. 利用Charbonnier loss代替L2作为惩罚函数的改进