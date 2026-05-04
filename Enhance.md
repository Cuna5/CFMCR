# EMRDM 改进记录 — Consistency Distillation（1步去云）

## 改进目标

基于论文 *Consistency Models*（Song et al., ICML 2023）和 *Consistency Flow Matching*（Yang et al., 2024），在 EMRDM（CVPR 2025）的均值回归扩散框架上引入 **一致性蒸馏（Consistency Distillation, CD）**，将推理步数从 4–5 步压缩至 **1 步**，同时保留原有的均值回归特性。

---

## 核心思路

**一致性条件**：对模型 $f_\theta$ 施加约束，使得沿同一 ODE 轨迹上的任意点，模型输出均应收敛到同一终点（无云图像）：

$$f_\theta(x_{\sigma_n}, \sigma_n, \mu) \approx f_{\theta^-}(x_{\sigma_{n-1}}, \sigma_{n-1}, \mu)$$

其中：
- $x_{\sigma_{n-1}}$ 由冻结的 **ODE teacher** $\phi$（从预训练 EMRDM checkpoint 复制）对 $x_{\sigma_n}$ 做一步 Euler ODE 积分得到
- $f_{\theta^-}$ 为 stop-gradient 的 EMA target network 预测（目标），不参与反向传播
- $f_\theta$ 为 student 预测，接收梯度更新

**训练时**：随机采样连续噪声对 $(\sigma_n, \sigma_{n-1})$，用冻结的预训练 ODE teacher 生成下一轨迹点，再用 EMA target network 生成一致性目标，最小化 student/target 预测间的 L2 距离。
**推理时**：仅调用一次 $f_\theta(x_{\sigma_{\max}}, \sigma_{\max}, \mu)$ 直接得到无云图像。

---

## 修改文件列表

### 1. `sgm/modules/diffusionmodules/loss.py`

**新增类** `ConsistencyResidualDiffusionLoss`（在 `TemporalResidualDiffusionLoss` 末尾后追加）

**主要逻辑**：

| 步骤 | 操作 |
|------|------|
| 1 | 从 EDM 离散化时间表采样随机连续对 $(\sigma_n, \sigma_{n-1})$ |
| 2 | 构造带噪图像：$x_{\sigma_n} = x_{clean} + \frac{1-s_n}{s_n}\mu + \sigma_n\varepsilon$ |
| 3 | 调用 `teacher_fn`（无梯度），得到 $x_{\sigma_{n-1}}$ 和 teacher 目标 $f_{target}$ |
| 4 | Student 在 $\sigma_n$ 处预测 $f_{student}$ |
| 5 | 损失 $\mathcal{L} = \|f_{student} - f_{target}\|_2^2$ |

**构造函数参数**：
```python
ConsistencyResidualDiffusionLoss(
    discretization_config,   # sigma 时间表配置
    loss_type="l2",          # "l2" 或 "l1"
    num_steps=18,            # 训练时采样 sigma 对的时间表步数
    batch2model_keys=None,
)
```

---

### 2. `sgm/modules/diffusionmodules/sampling.py`

**新增类** `ConsistencyResidualSampler`（在文件末尾 `IdealSampler.__call__` 后追加）

**推理流程**（仅 1 步）：
1. 从 EDM 离散化时间表取 $\sigma_{\max}$
2. 初始化：$x_{init} = \frac{1-s_{\max}}{s_{\max}}\mu + \sigma_{\max} \cdot \text{noise}$
3. 调用一次 denoiser：`denoised = denoiser(x_init, σ_max, cond, st_max)`
4. 直接返回 `denoised`

**调用接口**与 `ResidualEulerEDMSampler` 完全兼容（相同签名），无需改动 engine 中的 `sample()` 方法。

---

### 3. `sgm/models/diffusion.py`

**新增类** `ConsistencyResidualDiffusionEngine`（在 `ResidualDiffusionEngine` 与 `TemporalResidualDiffusionEngine` 之间插入）

继承自 `ResidualDiffusionEngine`，主要差异：

| 方法 | 说明 |
|------|------|
| `__init__` | 额外创建冻结的 `self.ode_teacher_model`（预训练 ODE 求解器）和 `self.teacher_model`（EMA target network） |
| `_update_teacher()` | EMA 更新 teacher：$\theta^- \leftarrow d \cdot \theta^- + (1-d) \cdot \theta$，默认衰减率 $d=0.9999$ |
| `on_train_batch_end()` | 在父类 EMA 更新后额外调用 `_update_teacher()` |
| `_make_teacher_fn()` | 返回 `@torch.no_grad()` 闭包：先用 frozen ODE teacher 执行 Euler ODE 步骤，再用 EMA target network 返回 $(x_{\sigma_{n-1}}, f_{target})$ |
| `forward()` | 构造 `teacher_fn` 并传给 `ConsistencyResidualDiffusionLoss` |

**构造函数新增参数**：
```python
ConsistencyResidualDiffusionEngine(
    teacher_ema_decay=0.9999,   # teacher EMA 衰减率
    *args, **kwargs             # 其余参数与 ResidualDiffusionEngine 相同
)
```

> **注意**：`use_ema` 被强制设为 `True`（验证/采样时使用 model_ema 权重）。CD 训练应加载预训练 EMRDM checkpoint；否则 frozen ODE teacher 只是随机初始化，蒸馏目标没有可靠意义。

---

### 4. `configs/example_training/cuhk_consistency.yaml`（新建）

基于 `cuhk.yaml` 修改的一致性蒸馏训练配置，关键差异：

| 字段 | 原值 | 新值 |
|------|------|------|
| `model.target` | `ResidualDiffusionEngine` | `ConsistencyResidualDiffusionEngine` |
| `model.params.base_learning_rate` | `1e-4` | `2e-5`（蒸馏微调使用更小 LR） |
| `model.params.teacher_ema_decay` | — | `0.9999` |
| `loss_fn_config.target` | `ResidualDiffusionLoss` | `ConsistencyResidualDiffusionLoss` |
| `loss_fn_config.params.num_steps` | — | `18` |
| `sampler_config.target` | `ResidualEulerEDMSampler` | `ConsistencyResidualSampler` |
| `max_epochs` | `500` | `200` |

---

## 文件变更总览

```
EMRDM-ODE/
├── sgm/
│   ├── modules/
│   │   └── diffusionmodules/
│   │       ├── loss.py          ← 新增 ConsistencyResidualDiffusionLoss
│   │       └── sampling.py      ← 新增 ConsistencyResidualSampler
│   └── models/
│       └── diffusion.py         ← 新增 ConsistencyResidualDiffusionEngine
└── configs/
    └── example_training/
        └── cuhk_consistency.yaml  ← 新建配置文件
```

---

## 改进效果预期

| 指标 | 原 EMRDM (4步) | Consistency Distillation (1步) |
|------|----------------|-------------------------------|
| 推理步数 | 4–5 步 | **1 步** |
| 推理速度 | 1× | **~4–5× 加速** |
| 训练方式 | 从零训练 | 在预训练 checkpoint 上蒸馏微调 |
| 质量损失 | — | 轻微（可通过 multi-step CD 弥补） |

---

---

# 改进记录 — Flow Matching（条件流匹配去云）

## 改进目标

基于论文 *Flow Matching for Generative Modeling*（Lipman et al., 2022）和 *Consistency Flow Matching*（Yang et al., 2024），在 EMRDM 框架上引入 **条件流匹配（Conditional Flow Matching, CFM）**，以直线 ODE 轨迹替代扩散过程，提升训练稳定性和推理效率。

> **实现策略**：所有新代码均以独立新文件形式提供（`*_fm.py`），**不修改**任何已有文件（包括 CD 改进已修改的文件）。`ResidualDiffusionEngine` 可直接复用，无需新 engine 类。

---

## 核心思路

**最优传输条件流**：定义从"有云图像 $\mu$"到"无云图像 $x_1$"的确定性直线路径：

$$x_t = (1-t)\,\mu + t\,x_1, \quad t \in [0, 1]$$

对应速度场（常数）：

$$v^* = x_1 - \mu$$

网络学习预测这个速度场，训练目标为：

$$\mathcal{L}_{FM} = \mathbb{E}_{t,x_1,\mu}\left[\|v_\theta(x_t, t, \mu) - (x_1 - \mu)\|^2\right]$$

**时间变量约定**（与 EMRDM 的 $\sigma$ 参数对齐）：
$$\sigma = 1 - t, \quad \sigma \in [0, 1]$$

推理时从 $\sigma \approx 1$（$t \approx 0$，初始点 $\approx \mu$）积分到 $\sigma \approx 0$（$t \approx 1$，无云图像），使用标准 Euler 方法。

---

## 新建文件列表

### 1. `sgm/modules/diffusionmodules/sigma2st_fm.py`（新建）

定义 FM 的时间尺度函数：$s(\sigma) = 1 - \sigma = t$

| 类 | 说明 |
|----|------|
| `FlowMatchingSigma2St` | 继承 `Sigma2St`；`s(σ) = 1 − σ`，导数 $ds/d\sigma = -1$（常数） |

```python
class FlowMatchingSigma2St(Sigma2St):
    def __call__(self, sigma):
        return 1.0 - sigma                              # s = t = 1 − σ
    def get_derivative_st(self):
        return lambda sigma: -torch.ones_like(sigma)   # ds/dσ = -1
```

**用途**：仅为 denoiser scaling 提供时间嵌入（`c_noise`），不参与 loss 中的路径计算。

---

### 2. `sgm/modules/diffusionmodules/denoiser_scaling_fm.py`（新建）

定义速度预测的网络预处理/后处理系数（velocity-prediction preconditioning）：

| 系数 | 取值 | 含义 |
|------|------|------|
| `c_skip` | $0$ | 网络输出不加残差连接 |
| `c_out` | $1$ | 网络输出直接作为速度 |
| `c_in` | $1$ | 输入不做缩放 |
| `c_noise` | $0.25 \log t$ | 时间步嵌入（$t = 1 - \sigma$） |

```python
class FlowMatchingScaling(DenoiserScaling):
    def __call__(self, sigma, st=None):
        c_skip  = torch.zeros_like(sigma)
        c_out   = torch.ones_like(sigma)
        c_in    = torch.ones_like(sigma)
        t       = (1.0 - sigma).clamp(min=1e-4)
        c_noise = 0.25 * torch.log(t)
        return c_skip, c_out, c_in, c_noise
```

> `c_skip=0` 意味着 denoiser 输出 = 网络输出（即预测的速度场 $v$），而非 $x_{clean}$。

---

### 3. `sgm/modules/diffusionmodules/loss_fm.py`（新建）

条件流匹配训练损失：

**主要逻辑**：

| 步骤 | 操作 |
|------|------|
| 1 | 从 $[t_{min}, t_{max}]$ 均匀采样 $t$，令 $\sigma = 1 - t$ |
| 2 | 构造插值路径：$x_t = (1-t)\mu + t \cdot x_{clean}$（无噪声，纯 OT 插值） |
| 3 | 计算速度目标：$v^* = x_{clean} - \mu$（常数，不依赖 $t$） |
| 4 | 通过 denoiser 预测速度：$v_{pred} = \text{denoiser}(x_t, \sigma, \text{cond}, s_t)$ |
| 5 | 损失 $\mathcal{L} = \|v_{pred} - v^*\|_2^2$ |

**构造函数参数**：
```python
FlowMatchingResidualDiffusionLoss(
    t_min=0.001,        # 最小时间步
    t_max=0.999,        # 最大时间步
    loss_type="l2",     # "l2" 或 "l1"
    batch2model_keys=None,
)
```

**与 `ResidualDiffusionLoss` 的签名对比**：`forward()` 参数完全相同（去掉 `teacher_fn`），因此可直接被 `ResidualDiffusionEngine.forward()` 调用。

---

### 4. `sgm/modules/diffusionmodules/sampling_fm.py`（新建）

FM Euler ODE 采样器（推理时使用）：

**推理流程**（$N$ 步 Euler）：
1. 初始化：$x_0 = \mu$（从有云图像出发，对应 $t \approx 0$）
2. 对 $(\sigma_i, \sigma_{i+1})$ 逐步积分：
   - $v_i = \text{denoiser}(x_i, \sigma_i, \text{cond}, s_i)$
   - $dt = \sigma_{i+1} - \sigma_i$（负数，因为 $\sigma$ 从大到小）
   - $x_{i+1} = x_i + (-v_i) \cdot dt = x_i - v_i \cdot dt$
3. 返回最终 $x$（$\approx x_{clean}$）

```python
class FlowMatchingResidualSampler:
    def __call__(self, denoiser, x, mu, cond, uc=None, num_steps=None, ...):
        x = mu.clone()   # 忽略 engine 传入的噪声，从 mu 出发
        sigmas = get_sigmas(...)  # 线性从 σ_max→σ_min
        for i, sigma in enumerate(sigmas[:-1]):
            v = denoiser(x, sigma, cond, st)
            dt = sigmas[i+1] - sigma   # dt < 0
            x = x - v * dt             # Euler step
        return x, ...
```

> `FlowMatchingResidualSampler` 与 `ConsistencyResidualSampler` 签名兼容，可相互替换用于消融实验。

---

### 5. `configs/example_training/cuhk_fm.yaml`（新建）

Flow Matching 在 CUHK-CR1 数据集上的训练配置：

| 字段 | 取值 | 说明 |
|------|------|------|
| `model.target` | `ResidualDiffusionEngine` | 直接复用，无需新 engine |
| `denoiser_scaling_config.target` | `FlowMatchingScaling` | 速度预测预处理 |
| `sigma2st_config.target` | `FlowMatchingSigma2St` | s(σ) = 1-σ |
| `loss_fn_config.target` | `FlowMatchingResidualDiffusionLoss` | CFM 损失 |
| `sampler_config.target` | `FlowMatchingResidualSampler` | Euler ODE 采样 |
| `sigma_max` | `0.999` | 对应 t_min ≈ 0.001 |
| `sigma_min` | `0.001` | 对应 t_max ≈ 0.999 |
| `rho` | `1` | EDMDiscretization 的 rho=1 → 线性 σ 时间表 |
| `num_steps` | `4` | Euler 积分步数（可按质量需求调大） |

---

### 6. `configs/example_training/cuhkv2_fm.yaml`（新建）

同上，针对 CUHK-CR2 多时相数据集调整：

| 差异字段 | 取值 |
|----------|------|
| `batch_size` | `1` |
| `check_val_every_n_epoch` | `5` |
| `max_epochs` | `2000` |
| `data.target` | `sgm.data.cuhk.image_datasets.TrainDataset`（路径切到 CUHK-CR2） |

---

## 文件变更总览

```
EMRDM-ODE/
├── sgm/
│   └── modules/
│       └── diffusionmodules/
│           ├── sigma2st_fm.py             ← 新建（FlowMatchingSigma2St）
│           ├── denoiser_scaling_fm.py     ← 新建（FlowMatchingScaling）
│           ├── loss_fm.py                 ← 新建（FlowMatchingResidualDiffusionLoss）
│           └── sampling_fm.py             ← 新建（FlowMatchingResidualSampler）
└── configs/
    └── example_training/
        ├── cuhk_fm.yaml                   ← 新建训练配置
        └── cuhkv2_fm.yaml                 ← 新建训练配置（多时相）
```

> 以上 FM 文件为新增实现，仍与原 EMRDM 主流程解耦。

---

## 与 EMRDM 原方法及 CD 改进的对比

| 维度 | EMRDM (原始) | Consistency Distillation | Flow Matching |
|------|-------------|--------------------------|---------------|
| ODE 轨迹 | 均值回归曲线 | 均值回归曲线（蒸馏） | **直线（OT 最优传输）** |
| 路径噪声 | 高斯噪声扰动 | 高斯噪声扰动 | **无噪声** |
| 网络预测目标 | $x_{clean}$ | $x_{clean}$（一致性约束） | **速度场 $v = x_1 - \mu$** |
| 训练推理步数 | 4–5 步 | **1 步** | 4–10 步（可减少） |
| 训练复杂度 | 基础 | 需要 teacher EMA + ODE 步骤 | **简单**（纯回归） |
| 对预训练依赖 | 从零训练 | 需要预训练 checkpoint | **从零训练** |
| 主要优势 | 精度高 | 推理极速 | 训练稳定、收敛快 |

---

---

# 改进记录 — Consistency Flow Matching（一步端点/速度一致性去云）

## 改进目标

综合 *Consistency Models*（Song et al., ICML 2023）与 *Flow Matching*（Lipman et al., 2022）的核心思想，在 EMRDM 框架上实现 **Consistency Flow Matching（CFM）**：在直线 OT 路径上施加端点一致性和速度自一致性约束，**从零训练**即可实现 **1 步去云**。

> **实现策略**：CFM 初版以独立新文件形式提供（`*_cfm.py`、`diffusion_cfm.py`）。本次校正直接更新这些 CFM 文件与配置，使实现更贴近论文公式。

---

## 核心思路

**CFM 的核心约束**：对于同一 OT 轨迹上的任意两个时刻 $t_n < t_{n+1}$，不仅要求速度场一致，还要求由速度外推到终点的预测一致：

$$f_\theta(t, x_t, \mu) = x_t + (1-t)\,v_\theta(x_t,t,\mu)$$

端点一致性：

$$f_\theta(t_n,x_{t_n},\mu) \approx f_{\theta^-}(t_{n+1},x_{t_{n+1}},\mu)$$

速度一致性：

$$v_\theta(x_{t_n}, t_n, \mu) \approx v_{\theta^-}(x_{t_{n+1}}, t_{n+1}, \mu)$$

其中 $v_{\theta^-}$ 是 EMA teacher 的预测（stop-gradient）。为避免从零训练时 student/teacher 只学到“自洽但错误”的速度场，额外加入 Flow Matching 的真实速度锚点：

$$v^* = x_{clean} - \mu$$

最终损失为：

$$\mathcal{L}_{CFM} =
\lambda_f\|f_{student}-f_{target}\|_2^2
+ \lambda_v\|v_{student} - v_{target}\|_2^2
+ \lambda_{FM}\|v_{student} - (x_{clean}-\mu)\|_2^2$$

**与 CD 和 FM 的核心差异**：

| | Flow Matching (Dir 1) | CD (Dir 2) | **CFM (Dir 3)** |
|---|---|---|---|
| 训练点 $x_{t_n}$ | 确定性 OT 公式 | EMRDM 随机噪声路径 | **确定性 OT 公式** |
| 一致性目标 | 无 | $x_{clean}$（图像空间） | **端点 $f$ + 速度场 $v$** |
| teacher 计算量 | 无 | ODE 步骤 + 2次 teacher 调用 | **1次 teacher 调用** |
| 1 步推理 | ✗（需多步） | ✓ | **✓** |
| 需预训练 | ✗ | ✓ | **✗** |

**关键优势**：因 OT 路径完全确定（已知 $x_{clean}$ 和 $\mu$ → 可精确计算 $x_{t_{n+1}}$），teacher 仅需在 $x_{t_{n+1}}$ 处做一次前向，**无需 ODE 模拟**，训练效率优于 CD。

---

## 新建文件列表

### 1. `sgm/modules/diffusionmodules/sigma2st_cfm.py`（新建）

`ConsistencyFlowMatchingSigma2St`，继承 `FlowMatchingSigma2St`（直接 alias），仅提供独立的类名以便 YAML 配置明确标识方向。

$$s(\sigma) = 1 - \sigma = t, \quad \frac{ds}{d\sigma} = -1$$

---

### 2. `sgm/modules/diffusionmodules/denoiser_scaling_cfm.py`（新建）

`ConsistencyFlowMatchingScaling`，继承 `FlowMatchingScaling`（直接 alias）。

| 系数 | 取值 |
|------|------|
| `c_skip` | $0$（网络输出 = 速度） |
| `c_out` | $1$ |
| `c_in` | $1$ |
| `c_noise` | $0.25\log t$，$t = 1 - \sigma$ |

---

### 3. `sgm/modules/diffusionmodules/loss_cfm.py`（新建）

**`ConsistencyFlowMatchingLoss`**，CFM 核心训练损失：

| 步骤 | 操作 |
|------|------|
| 1 | 从线性 σ 时间表采样连续对 $(\sigma_n, \sigma_{n+1})$（$\sigma_n > \sigma_{n+1}$） |
| 2 | **精确**构造 OT 路径训练点：$x_{t_n} = (1-t_n)\mu + t_n x_{clean}$ |
| 3 | **精确**构造下一步：$x_{t_{n+1}} = (1-t_{n+1})\mu + t_{n+1} x_{clean}$（无 ODE 模拟！） |
| 4 | Teacher（EMA）在 $x_{t_{n+1}}$ 处预测速度 $v_{target}$（**仅 1 次**调用，无梯度） |
| 5 | Student 在 $x_{t_n}$ 处预测速度 $v_{student}$ |
| 6 | 构造端点预测：$f_{student}=x_{t_n}+(1-t_n)v_{student}$，$f_{target}=x_{t_{n+1}}+(1-t_{n+1})v_{target}$ |
| 7 | 计算真实速度锚点 $v^* = x_{clean} - \mu$ |
| 8 | 损失 $\mathcal{L} = \lambda_f\|f_{student}-f_{target}\|_2^2 + \lambda_v\|v_{student} - v_{target}\|_2^2 + \lambda_{FM}\|v_{student} - v^*\|_2^2$ |

**构造函数参数**：
```python
ConsistencyFlowMatchingLoss(
    discretization_config,  # 线性 σ 时间表（rho=1）
    loss_type="l2",
    num_steps=18,           # 训练时 σ 对的时间表密度
    endpoint_loss_weight=1.0,
    consistency_loss_weight=1.0,  # 速度一致性
    fm_loss_weight=1.0,      # 真实速度监督锚点
    batch2model_keys=None,
)
```

**`forward()` 签名**（与 `ConsistencyResidualDiffusionLoss` 相同，第 2 参数为 `teacher_fn`）：
```python
forward(self, network, teacher_fn, denoiser, conditioner, sigma2st, input, mu, batch)
```

---

### 4. `sgm/modules/diffusionmodules/sampling_cfm.py`（新建）

**`ConsistencyFlowMatchingSampler`**，继承 `FlowMatchingResidualSampler`，新增 `_one_step` 方法：

**1 步推理**（`num_steps=1`）：

$$x_{init} = \mu, \quad v = v_\theta(\mu, \sigma_{\max}, \text{cond}), \quad x_{clean} = \mu + v$$

这里 $\sigma_{\max}$ 仅用于给网络提供接近 $t=0$ 的时间条件；输出端采用完整的 $t=0 \rightarrow t=1$ 一步一致性跳转，避免因为 $\sigma_{\max}=0.999$ 而少走最后的 $0.1\%$ 路径。

**多步推理**（`num_steps>1`）：自动委托父类 `FlowMatchingResidualSampler` 的 Euler 循环。

```python
class ConsistencyFlowMatchingSampler(FlowMatchingResidualSampler):
    def __call__(self, ...):
        if n == 1:  return self._one_step(...)   # 1-step
        else:       return super().__call__(...)  # multi-step Euler
```

---

### 5. `sgm/models/diffusion_cfm.py`（新建）

**`ConsistencyFlowMatchingEngine`**，继承 `ResidualDiffusionEngine`，主要差异：

| 方法 | 说明 |
|------|------|
| `__init__` | 深拷贝 student 创建 `self.teacher_model`（参数冻结） |
| `_update_teacher()` | EMA 更新：$\theta^- \leftarrow d\cdot\theta^- + (1-d)\cdot\theta$ |
| `on_train_batch_end()` | 调用父类后再调 `_update_teacher()` |
| `_make_teacher_fn()` | 返回轻量 `@torch.no_grad()` 闭包，仅包装 teacher denoiser |
| `forward()` | 注入 `teacher_fn` 作为 `loss_fn` 的第 2 参数 |

**CFM teacher_fn 比 CD teacher_fn 更轻量**：CD 需执行 EMRDM Euler 步骤（2 次 teacher 调用），CFM 的 teacher_fn 只做 1 次前向（位置由 OT 公式确定）：

```python
@torch.no_grad()
def teacher_fn(x, sigma, st, cond, **extra):
    return denoiser(teacher, x, sigma, cond, st, **extra).detach()
```

---

### 6. `configs/example_training/cuhk_cfm.yaml`（新建）

CFM 在 CUHK-CR1 上的训练配置：

| 字段 | 取值 | 说明 |
|------|------|------|
| `model.target` | `ConsistencyFlowMatchingEngine` | CFM 引擎（`diffusion_cfm.py`） |
| `teacher_ema_decay` | `0.9999` | EMA teacher 衰减率 |
| `sigma_st_config.target` | `ConsistencyFlowMatchingSigma2St` | s(σ) = 1-σ |
| `denoiser_scaling_config.target` | `ConsistencyFlowMatchingScaling` | c_skip=0，速度预测 |
| `loss_fn_config.target` | `ConsistencyFlowMatchingLoss` | CFM 端点 + 速度一致性损失 |
| `loss_fn_config.params.endpoint_loss_weight` | `1.0` | 端点一致性损失权重 |
| `loss_fn_config.params.num_steps` | `18` | σ 时间表密度 |
| `sampler_config.target` | `ConsistencyFlowMatchingSampler` | 1 步采样 |
| `sampler_config.params.num_steps` | `1` | 推理仅 1 步 |
| `rho` | `1` | 线性 σ 时间表 |
| `max_epochs` | `500` | 从零训练 |

---

### 7. `configs/example_training/cuhkv2_cfm.yaml`（新建）

同上，针对 CUHK-CR2 多时相数据集：

| 差异字段 | 取值 |
|----------|------|
| `batch_size` | `1` |
| `check_val_every_n_epoch` | `5` |
| `max_epochs` | `2000` |

---

## 文件变更总览

```
EMRDM-ODE/
├── sgm/
│   ├── models/
│   │   └── diffusion_cfm.py              ← 新建（ConsistencyFlowMatchingEngine）
│   └── modules/
│       └── diffusionmodules/
│           ├── sigma2st_cfm.py            ← 新建（ConsistencyFlowMatchingSigma2St）
│           ├── denoiser_scaling_cfm.py    ← 新建（ConsistencyFlowMatchingScaling）
│           ├── loss_cfm.py                ← 新建（ConsistencyFlowMatchingLoss）
│           └── sampling_cfm.py            ← 新建（ConsistencyFlowMatchingSampler）
└── configs/
    └── example_training/
        ├── cuhk_cfm.yaml                  ← 新建训练配置
        └── cuhkv2_cfm.yaml                ← 新建训练配置（多时相）
```

> 这些是 CFM 初版新增文件；本次校正已在 `loss_cfm.py`、`sampling_cfm.py`、`diffusion_cfm.py` 与 CFM YAML 中补齐论文对应的端点一致性和一步推理端点处理。

---

## 四种方法完整对比

| 维度 | EMRDM (原始) | CD (Dir 2) | FM (Dir 1) | **CFM (Dir 3)** |
|------|-------------|------------|------------|-----------------|
| ODE 轨迹 | 均值回归曲线 | 均值回归（蒸馏） | 直线 OT | **直线 OT** |
| 训练点噪声 | 高斯噪声 | 高斯噪声 | 无噪声 | **无噪声** |
| 网络预测目标 | $x_{clean}$ | $x_{clean}$ | 速度 $v$ | **速度 $v$** |
| 1 步推理 | ✗ | ✓ | ✗ | **✓** |
| 需要预训练 | 否 | **是** | 否 | **否** |
| teacher 计算量 | 无 | 2 次调用 + ODE | 无 | **1 次调用** |
| 训练复杂度 | 基础 | 中等 | 简单 | **中等** |
| 主要优势 | 精度高 | 推理最快 | 收敛稳定 | **1步+无需预训练** |

---

# 运行耗时统计补充

## 改进目标

为方便比较 EMRDM、CD、FM、CFM 的实际效率，在 `ResidualDiffusionEngine` 中加入可选计时开关：

| 参数 | 记录指标 | 说明 |
|------|----------|------|
| `count_sample_time=True` | `sample_time` | 推理耗时，覆盖 validation/test/predict 的 sample + decode 阶段 |
| `count_train_time=True` | `train_time`、`train_time_avg` | 训练 batch 耗时和运行均值，不包含 dataloader 取数时间 |

## 实现思路

- CUDA 环境使用 `torch.cuda.Event`，保证 GPU 异步执行完成后再统计。
- 非 CUDA 环境自动回退到 `time.perf_counter()`，避免 CPU/MPS 调试时报错。
- `--test` / validation 中的 `sample_time` 同步写入 Lightning logger；`--predict` 中的 `sample_time` 写入 `metrics.csv`。
- CD/CFM 子类的 teacher EMA 更新被纳入 `train_time`，因此训练耗时更接近完整 batch 成本。
