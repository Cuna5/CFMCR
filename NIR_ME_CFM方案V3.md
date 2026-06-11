# NIR-ME-CFM 方案 V3：保留 V1 创新叙事的可落地版本

方法名：

```text
NIR-ME-CFM:
NIR-derived Multi-scale Edge-preserving Consistency Flow Matching
for Remote Sensing Cloud Removal
```

中文名称：

```text
近红外派生多尺度边缘保持一致性流匹配遥感图像去云方法
```

V3 的目标是：**尽量保留 V1 的完整方法叙事和三个创新点，同时吸收 V2 中更稳的工程约束**。因此 V3 不把方案缩成一个纯工程补丁，而是保留：

- `NCB`: NIR-guided Conditional Backbone
- `TMM`: Time-aware Multimodal Modulation
- `MEF`: Multi-scale Edge-preserving Fusion

但在实现上避免以下风险：

- 不改 CFM 的 loss / sampler / denoiser / Engine
- 不重新引入旧 EMRDM 的 mean shifting / 加噪前向逻辑
- 不把 label-derived `M` 作为网络输入
- 不在高分辨率层默认使用全局 cross-attention
- 不破坏现有 `in_channels=8` 和 CFM checkpoint 热启动

---

## 0. 和当前 CFM 的关系

当前 CFM 中：

```text
mu = cond_image = x_cloudy
input = label = x_clean
```

训练路径：

```text
x_t = sigma * x_cloudy + (1 - sigma) * x_clean
v_target = x_clean - x_cloudy
```

一步推理：

```text
x_pred = x_cloudy + sigma_max * v_theta(x_cloudy, sigma_max, cond)
```

因此 V3 只增强神经网络如何利用条件信息，不改变 CFM 定义。

不动文件：

```text
sgm/models/diffusion_cfm.py
sgm/modules/diffusionmodules/loss_cfm.py
sgm/modules/diffusionmodules/sampling_cfm.py
sgm/modules/diffusionmodules/sigma2st_cfm.py
sgm/modules/diffusionmodules/denoiser_scaling_cfm.py
```

---

## 1. 整体框架

V1 的整体思想保留：CFM 提供恢复路径，NIR / NDVI / Edge 提供遥感结构先验。

```text
Input:
  x_t                         CFM 当前状态
  cloudy RGBNIR x_cloudy      含云条件图，也就是 mu / cond_image

Auxiliary cues from x_cloudy:
  NDVI_cloudy
  Edge_RGB
  Edge_NIR

Backbone:
  NCB: NIR-guided Conditional Backbone
  TMM: Time-aware Multimodal Modulation
  MEF: Multi-scale Edge-preserving Fusion

CFM head:
  v_theta(x_t, t, condition)

Output:
  clear RGBNIR x_pred
```

更贴近当前代码的结构：

```text
cloudy RGBNIR x_cloudy
   ├─ 作为 CFM 路径起点: mu
   ├─ 作为 concat 条件: cat(x_t, x_cloudy) -> 8ch
   └─ 派生 aux_cond:
        NDVI_cloudy + Edge_RGB + Edge_NIR -> 5ch

cat(x_t, x_cloudy) [B,8,H,W]
   -> patch_in
   -> Hourglass Transformer
        ├─ NCB: aux_global 注入 Mapping cond
        ├─ TMM: 根据 CFM 时间步调制 NDVI / Edge 贡献
        └─ MEF: 多尺度边缘空间注入
   -> velocity v_theta
```

这里的关键点是：**主输入仍然是 8 通道，aux_cond 不拼入主输入，而是作为额外 keyword 进入网络。**

---

## 2. 辅助条件构造

### 2.1 只使用推理时可获得的信息

辅助条件只从含云 RGBNIR 构造：

```text
cond_image: [B,4,H,W]
channels:   R, G, B, NIR
range:      [-1,1]
```

先转回 `[0,1]`：

```python
rgbnir = (cond_image.clamp(-1, 1) + 1.0) * 0.5
red = rgbnir[:, 0:1]
rgb = rgbnir[:, 0:3]
nir = rgbnir[:, 3:4]
```

构造：

```python
ndvi = (nir - red) / (nir + red + 1e-4)
edge_rgb = sobel(rgb)
edge_nir = sobel(nir)
aux_cond = torch.cat([ndvi, edge_rgb, edge_nir], dim=1)
```

最终：

```text
aux_cond: [B,5,H,W]
  channel 0: NDVI_cloudy
  channel 1: Edge_R
  channel 2: Edge_G
  channel 3: Edge_B
  channel 4: Edge_NIR
```

### 2.2 关于云掩膜 M

当前 CUHK dataloader 里的 `M` 是：

```python
M = np.clip((t - x).sum(axis=2), 0, 1)
```

这里的 `t` 是 label，因此这个 `M` 不能作为网络输入，否则会真值泄漏。

V3 约定：

```text
M 可以继续用于 loss weighting；
M 不进入 NCB / TMM / MEF；
若未来有推理时可用 cloud_prob，再单独做 cloud-aware aux 版本。
```

如果后续有可推理云概率图：

```python
clear_weight = 1.0 - cloud_prob
ndvi = ndvi * clear_weight
edge_rgb = edge_rgb * clear_weight
edge_nir = edge_nir * clear_weight
```

---

## 3. 创新点 1：NCB，NIR-guided Conditional Backbone

### 3.1 保留 V1 的动机

RGB 图像在厚云区域结构信息不足，少步 CFM 容易出现：

```text
道路边缘弯曲
建筑轮廓融化
农田边界错位
水体岸线变形
植被区域光谱不稳定
```

NIR 和 NDVI 对植被、水体、农田边界等遥感结构更敏感；Edge maps 则显式提供几何边界。NCB 的目标是让这些信息不是只停留在输入拼接层，而是进入 backbone 的条件调制路径。

### 3.2 V3 的实现方式

V1 里写的是完整多分支 backbone：

```text
RGB branch
NIR branch
NDVI branch
Edge branch
Flow branch
Time branch
```

V3 保留这个概念，但工程实现先采用更稳的轻量版本：

```text
Flow branch:
  x_t + cloudy RGBNIR 仍然走原来的 8ch patch_in

Aux branch:
  aux_cond = NDVI + Edge_RGB + Edge_NIR
  AuxGlobalEncoder(aux_cond) -> aux_global
  cond' = cond + aux_global
```

也就是说，**不重写整个 Hourglass Transformer 的主干**，而是在 `MappingNetwork` 输出的 `cond` 上做 NIR-derived 条件注入。

### 3.3 NCB 公式

当前网络中每个 attention / FFN block 通过 `AdaRMSNorm(x, cond)` 接收条件。令：

```text
cond_t = Mapping(t)
F_aux = AuxGlobalEncoder(NDVI, Edge_RGB, Edge_NIR)
cond'_t = cond_t + W_aux(F_aux)
```

其中 `W_aux` 使用 zero-init：

```text
W_aux = 0 at init
```

所以热启动时：

```text
cond'_t = cond_t
```

模型行为与旧 CFM checkpoint 完全一致。

### 3.4 代码形态

```python
self.aux_global = nn.Sequential(
    nn.Conv2d(aux_channels, 64, 3, padding=1),
    nn.SiLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.SiLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
)
self.aux_to_cond = zero_init(Linear(64, mapping.width, bias=False))
```

```python
time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
cond = self.mapping(time_emb)

if self.use_aux_cond and aux_cond is not None:
    aux_feat = self.aux_global(aux_cond)
    cond = cond + self.aux_to_cond(aux_feat)
```

### 3.5 论文表述

可以保留 V1 的说法，但稍微修正：

> We propose a NIR-derived conditional backbone that explicitly extracts NDVI and multi-band edge cues from cloudy RGBNIR observations and injects them into the time-conditioned modulation pathway of the CFM backbone.

避免说成“新增 NIR 模态”，因为当前输入本身已经包含 NIR。

---

## 4. 创新点 2：TMM，Time-aware Multimodal Modulation

### 4.1 保留 V1 的动机

CFM 不同阶段对信息需求不同：

| CFM 时间阶段 | 主要任务 | 更需要的信息 |
|---|---|---|
| 接近含云图 | 建立整体结构、去除大块云 | RGB/NIR 全局结构、NDVI |
| 中间阶段 | 修正地物区域和云边界 | NIR、NDVI、Edge |
| 接近干净图 | 细化纹理和边界 | Edge_RGB、Edge_NIR |

如果所有时间步都固定强度使用 NDVI / Edge，可能出现：

```text
早期边缘过强，整体恢复不稳
后期边缘不足，边界模糊或漂移
```

### 4.2 当前 CFM 时间变量

代码中 CFM 使用：

```text
sigma = 1 - t
t = 1 - sigma
```

网络实际收到的 `timesteps` 是 CFM scaling 后的：

```python
c_noise = 0.25 * log(t)
```

因此 TMM 可以直接使用 `c_noise` 做门控输入。

### 4.3 V3 推荐实现

将 aux 分成两类：

```text
spectral cue: NDVI
edge cue: Edge_RGB + Edge_NIR
```

时间门控：

```text
[w_ndvi(t), w_edge(t)] = sigmoid(MLP(c_noise))
```

特征调制：

```text
F_aux(t) = concat(
    w_ndvi(t) * Enc_ndvi(NDVI),
    w_edge(t) * Enc_edge(Edge_RGB, Edge_NIR)
)
```

输出仍然 zero-init 到 `cond`：

```text
cond'_t = cond_t + zero_init(W)(F_aux(t))
```

### 4.4 代码形态

```python
class AuxGlobalEncoder(nn.Module):
    def __init__(self, mapping_width):
        super().__init__()
        self.ndvi_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.SiLU()
        )
        self.edge_enc = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.SiLU()
        )
        self.time_gate = nn.Sequential(
            Linear(1, 32), nn.SiLU(),
            Linear(32, 2),
            nn.Sigmoid(),
        )
        self.out_proj = zero_init(Linear(48, mapping_width, bias=False))

    def forward(self, aux_cond, c_noise):
        ndvi = aux_cond[:, 0:1]
        edge = aux_cond[:, 1:5]

        f_ndvi = self.ndvi_enc(ndvi)
        f_edge = self.edge_enc(edge)

        gates = self.time_gate(c_noise.unsqueeze(-1))
        w_ndvi = gates[:, 0:1, None, None]
        w_edge = gates[:, 1:2, None, None]

        f = torch.cat([f_ndvi * w_ndvi, f_edge * w_edge], dim=1)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.out_proj(f)
```

`out_proj` zero-init 已经保证热启动等价。若希望门控初始为等权，可把 `time_gate` 最后一层初始化为 0，使 sigmoid 后为 0.5。

### 4.5 论文表述

> We introduce a time-aware multimodal modulation module that adaptively balances spectral and edge cues according to the CFM time step, enabling the model to emphasize global spectral structure at early restoration stages and local boundary cues near the clean endpoint.

---

## 5. 创新点 3：MEF，Multi-scale Edge-preserving Fusion

### 5.1 保留 V1 的动机

少步 CFM 容易产生边缘漂移，因为模型需要在极少步内同时完成：

```text
去云
补纹理
恢复颜色
保持几何结构
恢复 NIR 光谱
```

如果 decoder 缺少显式边界提示，可能出现：

```text
道路断裂
水体岸线扭曲
建筑边缘融化
农田块状边界错位
```

MEF 的目标是保留 V1 中“在多尺度 skip / decoder 处注入边缘引导”的核心想法。

### 5.2 为什么不用 V1/V2 中的 full cross-attention 默认实现

原始 cross-attention 形式：

```text
Q = decoder tokens
K,V = aux edge tokens
Attention(Q,K,V)
```

表达力强，但当前 `patch_size=[1,1]`，高分辨率 token 数可能很大：

```text
128 x 128 = 16384 tokens
attention matrix = 16384 x 16384
```

这会带来显存和速度风险。因此 V3 保留 MEF 的思想，但默认实现改为轻量空间门控。

### 5.3 MEF-Lite：多尺度边缘门控残差

对每个 decoder 尺度：

```text
E_i = EdgeEncoder_i(aux_cond)
A_i = sigmoid(Gate_i(S_i, E_i))
S_i' = S_i + A_i ⊙ Phi_i(E_i)
```

其中：

- `S_i` 是 decoder 当前尺度特征；
- `E_i` 是同分辨率辅助边缘/NDVI 特征；
- `A_i` 是边缘注入强度；
- `S_i'` 是边缘增强后的 decoder 特征。

### 5.4 工程实现

简化实现可以先不用显式 `A_i`，直接 zero-init residual：

```python
class AuxSpatialGate(nn.Module):
    def __init__(self, width, aux_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(aux_channels, width, 3, padding=1),
            nn.SiLU(),
            zero_init(nn.Conv2d(width, width, 1)),
        )

    def forward(self, x, aux_cond):
        # x: [B,H,W,C]
        aux = F.interpolate(
            aux_cond,
            size=x.shape[1:3],
            mode="bilinear",
            align_corners=False,
        )
        aux = self.proj(aux).movedim(1, -1)
        return x + aux
```

插入点保留 V1/V2 的位置：

```python
for up_level, split, skip, pos in reversed(...):
    x = split(x, skip)
    if aux_spatial_mode == "gate":
        x = self.aux_spatial_gates[idx](x, aux_cond)
    x = up_level(x, pos, cond)
```

### 5.5 可选增强：低分辨率局部 Cross-Attention

如果 MEF-Lite gate 已经验证有收益，再尝试：

```text
只在 token 数 <= 4096 的 decoder 层启用 local/window cross-attention
最高分辨率层继续使用 AuxSpatialGate
```

这样保留 V1 中 cross-attention 的表达力，但避免高分辨率全局 attention 爆显存。

### 5.6 论文表述

> To reduce boundary drift in few-step CFM sampling, we inject NIR-derived edge cues into decoder features through a multi-scale edge-preserving fusion module. A lightweight gated residual design is adopted by default, while local cross-attention can be used at low-resolution stages for stronger spatial interaction.

---

## 6. 三个创新点如何组合

V1 的三模块结构保留：

```text
NCB: NIR-guided Conditional Backbone
TMM: Time-aware Multimodal Modulation
MEF: Multi-scale Edge-preserving Fusion
```

V3 的实际数据流：

```text
x_t, x_cloudy, sigma
│
├─ 主干输入:
│    cat(x_t, x_cloudy) -> [B,8,H,W]
│
├─ 辅助条件:
│    aux_cond = NDVI + Edge_RGB + Edge_NIR -> [B,5,H,W]
│
├─ NCB:
│    aux_cond -> aux_global -> cond' = cond + aux_global
│
├─ TMM:
│    c_noise -> modality gates -> time-aware aux_global
│
└─ MEF:
     aux_cond -> multi-scale spatial gate -> decoder features

Output:
  v_theta -> x_pred
```

模块关系：

| 模块 | 保留自 V1 的核心 | V3 的工程实现 |
|---|---|---|
| NCB | NIR / NDVI / Edge 条件增强 backbone | aux_global 注入 Mapping cond |
| TMM | 不同 CFM 阶段动态调节模态贡献 | 用 `c_noise` 控制 NDVI / Edge 门控 |
| MEF | 多尺度边缘保持，减少边缘漂移 | decoder 多尺度 zero-init spatial gate |

---

## 7. 接入方式

### 7.1 不让 aux_cond 进入 concat

当前 `GeneralConditioner` 会把 4D tensor 默认归到 `"concat"`。如果直接：

```yaml
target: sgm.modules.encoders.modules.IndentityEmbedder
input_key: "aux_cond"
```

会导致：

```text
cat(x_t, cond_image, aux_cond) = 13ch
```

这会破坏当前：

```yaml
network_config.params.in_channels: 8
```

因此新增：

```python
class DictEmbedder(AbstractEmbModel):
    def __init__(self, output_key):
        super().__init__()
        self.output_key = output_key

    def forward(self, x):
        return {self.output_key: x}
```

配置：

```yaml
conditioner_config:
  params:
    emb_models:
      - is_trainable: True
        input_key: "cond_image"
        ucg_rate: 0.0
        target: sgm.modules.encoders.modules.IndentityEmbedder

      - is_trainable: False
        input_key: "aux_cond"
        ucg_rate: 0.0
        target: sgm.modules.encoders.modules.DictEmbedder
        params:
          output_key: "aux_cond"
```

`CloudRemovalWrapper` 会把非 `"concat"` 的 key 传给网络：

```python
c_pass = {k: v for k, v in c.items() if k != "concat"}
return self.diffusion_model(x, timesteps=t, **c_pass, **kwargs)
```

网络 forward 增加：

```python
def forward(self, x, timesteps, aux_cond=None, control=None):
    ...
```

---

## 8. 配置建议

### 8.1 NCB only

```yaml
network_config:
  params:
    use_aux_cond: true
    aux_channels: 5
    aux_global_mode: "nce"
    aux_spatial_mode: "none"
```

### 8.2 NCB + TMM

```yaml
network_config:
  params:
    use_aux_cond: true
    aux_channels: 5
    aux_global_mode: "tam"
    aux_spatial_mode: "none"
```

### 8.3 Full V3

```yaml
network_config:
  params:
    use_aux_cond: true
    aux_channels: 5
    aux_global_mode: "tam"
    aux_spatial_mode: "gate"
    aux_spatial_max_tokens: 4096
```

推理保持确定性：

```yaml
sampler_config:
  params:
    num_steps: 1
    s_churn: 0.0
    tta: False
```

`tta=True` 只作为最终附加评估，不和结构收益混在一起。

---

## 9. 修改点清单

| 文件 | 修改内容 |
|---|---|
| `sgm/data/cuhk/image_datasets.py` | 从含云 RGBNIR 计算并返回 `aux_cond` |
| `sgm/modules/encoders/modules.py` | 新增 `DictEmbedder` |
| `sgm/modules/diffusionmodules/k_diffusion/image_transformer.py` | 增加 `use_aux_cond`、NCB/TMM/MEF-Lite 模块 |
| `configs/example_training/cuhk_cfm.yaml` | 增加 aux embedder 和网络参数 |
| `configs/example_training/cuhkv2_cfm.yaml` | 同步 CUHK-CR2 配置 |
| `sgm/modules/diffusionmodules/wrappers.py` | 原则上可不改；如需更显式地处理 `aux_cond` 可小改 |

不修改：

```text
loss / sampler / denoiser / Engine
```

---

## 10. 热启动与训练策略

### 10.1 热启动

从稳定 CFM checkpoint 微调：

```yaml
model:
  base_learning_rate: 2.0e-5
  params:
    ckpt_path: "/path/to/stable_cfm.ckpt"
```

新增输出层全部 zero-init：

```text
aux_to_cond
AuxGlobalEncoder.out_proj
AuxSpatialGate 最后一层 Conv
可选 local cross-attn 的 out_proj
```

加载旧 checkpoint 时 `strict=False`，旧参数正常加载，新参数从零开始。

### 10.2 阶段训练

| 阶段 | 配置 | 目的 |
|---|---|---|
| Stage 0 | Baseline CFM | 稳定基线 |
| Stage 1 | +NCB | 验证 NDVI/Edge 全局注入 |
| Stage 2 | +TMM | 验证时间感知调制 |
| Stage 3 | +MEF-Gate | 验证多尺度边缘空间注入 |
| Stage 4 | +MEF-Local | 低分辨率局部 cross-attn，可选 |

建议：

```text
先微调 5k-20k step 看趋势；
优先监控 RMSE / PSNR；
SSIM 作为结构辅助指标；
不要一开始就从零训练 Full V3。
```

---

## 11. 消融实验

| 实验 | 配置 | 验证点 |
|---|---|---|
| Baseline | 原始 CFM | 基线 |
| +AuxData | 只返回 aux_cond，网络不用 | 排除数据管线影响 |
| +NCB | aux_global 注入 cond | NIR-derived 条件是否有效 |
| +TMM | NCB + 时间门控 | 不同时间步动态模态权重是否有效 |
| +MEF-Gate | TMM + 空间门控残差 | 多尺度边缘注入是否改善边界 |
| +MEF-Local | 低分辨率 local/window cross-attn | 更强空间交互是否值得 |
| +TTA | 最优结构 + TTA | 推理增强收益 |

固定设置：

```text
sampler.num_steps = 1
s_churn = 0
同一 test split
同一 checkpoint 初始化
同一 image_metrics=evaluator
```

---

## 12. 风险与应对

| 风险 | 应对 |
|---|---|
| `M` 来自 label，作为输入会泄漏真值 | V3 默认不把 `M` 输入网络 |
| NDVI / Edge 在厚云区失真 | 作为弱条件注入，zero-init，小学习率微调 |
| 云边缘被 Sobel 当成地物边缘 | NCB 先全局注入；MEF-Gate 权重从零开始 |
| MEF full attention 爆显存 | 默认 gate；local attention 只在低分辨率启用 |
| 新模块破坏已收敛 CFM 速度场 | 所有新增输出 zero-init，分阶段微调 |
| 论文贡献被质疑只是手工特征 | 强调“派生遥感线索的时间感知、多尺度注入机制” |

---

## 13. V3 最终推荐实现

近期最推荐实现：

```text
NIR-ME-CFM V3:
  1. CFM flow / loss / sampler 不动
  2. mu 继续表示 cloudy RGBNIR
  3. aux_cond = NDVI_cloudy + Edge_RGB + Edge_NIR
  4. aux_cond 通过 DictEmbedder 作为 keyword 输入网络
  5. NCB: aux_global 注入 Mapping cond
  6. TMM: c_noise 控制 NDVI / Edge 动态权重
  7. MEF: decoder 多尺度 zero-init spatial gate
  8. local/window cross-attn 只作为后续增强
```

一句话总结：

```text
V3 保留 V1 的三创新点叙事：
NIR 条件骨干 + 时间感知调制 + 多尺度边缘保持；
同时采用 V2 的工程护栏：
不泄漏 label，不改 CFM，不破坏 8ch 输入，不默认高分辨率全局 attention。
```
