# 基于 CFM 的遥感图像去云性能提升方向总结

## 1. 总体思路

如果已经将原来的扩散模型替换为 CFM（Consistency Flow Matching），后续优化的核心不应该是继续大幅更换模型结构，而是把 CFM 改造成更适合遥感图像去云的 **PSNR-oriented restoration flow**。

换句话说，CFM 不应被当作普通生成模型来训练，而应被设计成：

> 从 cloudy image 精确回归到 cloudless image 的条件恢复模型。

核心目标是：

- 少生成，多回归；
- 少随机，多确定；
- 少追求视觉“合理”，多追求像素级一致；
- 保留 cloudy image 中已有的有效信息；
- 云区重点修复，非云区尽量不动。

---

## 2. 路径设计：从 noise → clear 改为 cloudy → clear

普通 CFM 往往学习的是：

```text
noise → clean image
```

但遥感图像去云任务更适合学习：

```text
cloudy image → cloudless image
```

推荐构造线性恢复路径：

```text
x_t = (1 - t) * x_cloudy + t * x_clear
```

对应的目标速度为：

```text
v_target = x_clear - x_cloudy
```

网络预测：

```text
vθ(t, x_t, condition)
```

最终输出：

```text
x_pred = x_t + (1 - t) * vθ(t, x_t, condition)
```

其中 `condition` 可以包含：

```text
cloudy image
cloud mask / cloud probability
SAR / IR image
multi-temporal images
time embedding
```

这样做的好处是，模型学习的是一条从退化图像到清晰图像的恢复路径，而不是从随机噪声生成图像，更有利于 PSNR。

---

## 3. 最终输出 MSE 应作为主损失

CFM 中常见的 velocity loss 是：

```text
L_v = ||vθ - v_target||²
```

但 PSNR 衡量的是最终输出图像和 ground truth 之间的 MSE，因此训练时必须直接优化最终输出：

```text
L_rec = MSE(x_pred, x_clear)
```

推荐基础损失：

```text
L_total =
1.0 * MSE(x_pred, x_clear)
+ 0.1 * MSE(vθ, v_target)
+ 0.05 * L_velocity_consistency
```

注意：

- 如果目标是提升 PSNR，不建议一开始加入较大的 perceptual loss、GAN loss 或 LPIPS loss；
- 这些损失可能提升视觉自然度，但也可能引入像素偏差，导致 PSNR 下降。

---

## 4. 加入 Non-cloud Identity Loss

这是最重要、最稳的提升方向之一。

遥感去云任务中，非云区域本来就是清晰的，模型不应该改动这些区域。由于 PSNR 是整图平均指标，如果非云区被轻微改坏，整体 PSNR 会明显下降。

设：

```text
M = 1 表示云区
M = 0 表示非云区
```

非云区保持损失为：

```text
L_id = MSE((1 - M) * x_pred, (1 - M) * x_cloudy)
```

加入后：

```text
L_total =
1.0 * MSE(x_pred, x_clear)
+ 0.2 * L_id
+ 0.1 * MSE(vθ, v_target)
+ 0.05 * L_velocity_consistency
```

作用：

- 防止非云区域被模型误修改；
- 保持原图结构和纹理；
- 对整图 PSNR 通常比较友好。

---

## 5. 加入 Cloud-weighted MSE

云区是模型真正需要修复的区域，因此可以对云区赋予更高权重：

```text
L_cloud = MSE((1 + αM) * (x_pred - x_clear))
```

建议初始设置：

```text
α = 1 或 2
```

不建议一开始将 `α` 设置得过大。因为 PSNR 是整图指标，如果过度关注云区，导致非云区或边缘区域变差，整体 PSNR 反而可能下降。

较稳的组合为：

```text
L_total =
1.0 * MSE(x_pred, x_clear)
+ 0.2 * L_id
+ 0.3 * L_cloud
+ 0.1 * L_v
+ 0.05 * L_consistency
```

---

## 6. 使用残差预测，而不是直接预测整图

对于云去除任务，直接预测整张无云图像可能导致模型修改过多区域。更推荐让模型预测残差：

```text
rθ = x_clear - x_cloudy
x_pred = x_cloudy + rθ
```

对应到 CFM 中，可以让 velocity 学习残差方向：

```text
v_target = x_clear - x_cloudy
```

非云区域的残差应该接近 0，因此可以加入：

```text
L_res_clear = MSE((1 - M) * rθ, 0)
```

作用：

- 降低学习难度；
- 强化非云区域保持；
- 减少不必要的纹理幻觉；
- 更符合 PSNR 优化目标。

---

## 7. 使用 Multi-segment CFM：优先尝试 2-step 或 4-step

如果目标是速度，1-step 很有吸引力；但如果目标是 PSNR，1-step 往往不够精确。

推荐实验：

```text
K = 1：1-step，速度最快，但 PSNR 可能不足
K = 2：2-step，速度和 PSNR 平衡
K = 4：4-step，更适合冲 PSNR
```

对于厚云区域，cloudy → clear 的映射可能并不是一条简单直线，因此分段线性 flow 通常比单段 flow 更稳定。

建议优先测试：

```text
K = 2 或 K = 4
```

采样方式建议使用 deterministic Euler，不加入随机噪声。

---

## 8. 使用 EMRDM 作为 Teacher 做蒸馏

如果已经有 EMRDM 或原扩散模型的较好结果，可以将其作为 teacher，帮助 CFM student 学习更稳定的恢复路径。

流程：

```text
x_teacher = EMRDM_5step(x_cloudy, condition)
x_student = CFM(x_cloudy, condition)
```

蒸馏损失：

```text
L_distill = MSE(x_student, x_teacher)
```

总损失：

```text
L_total =
1.0 * MSE(x_student, x_clear)
+ 0.2 * MSE(x_student, x_teacher)
+ 0.05 * L_velocity_consistency
```

注意：

- ground truth MSE 必须是主项；
- teacher loss 只能辅助；
- 如果过度模仿 teacher，student 的性能上限会被 teacher 限制。

更合理的理解是：

```text
teacher 指导路径稳定性
ground truth 指导最终 PSNR 上限
```

---

## 9. 加强条件输入

CFM 替换后容易出现“生成式脑补”的问题，因此需要增强条件约束。

推荐条件输入：

```text
condition =
cloudy image
+ cloud mask / cloud probability
+ SAR / IR
+ multi-temporal images
+ time embedding
```

其中最重要的是：

1. cloudy image：保留原始结构；
2. cloud mask：指导模型哪里该修，哪里不该修；
3. SAR / IR：在厚云区域提供辅助信息；
4. multi-temporal images：用其他时相补充被云遮挡区域。

如果没有真实 cloud mask，可以先使用：

- 简单阈值法；
- Fmask；
- 轻量云检测网络；
- 伪标签 cloud probability map。

即使 mask 不完美，也可以用于 loss weighting 和 identity loss。

---

## 10. 多时相任务：使用 Mask-guided Temporal Fusion

如果使用多时相数据，不建议简单平均多个时相结果。

原始 mean fusion：

```text
x_final = mean(x1, x2, x3)
```

可以改为 cloud-reliability weighted fusion：

```text
x_final = w1 * x1 + w2 * x2 + w3 * x3
```

权重可以由云概率决定：

```text
clear_score_l = 1 - cloud_prob_l
w_l = clear_score_l / sum(clear_score)
```

也可以使用可学习模块预测：

```text
w_l = FusionNet(cloud_mask_l, feature_l, attention_l)
```

作用：

- 避免云厚或质量差的时相污染最终结果；
- 提升厚云区域恢复质量；
- 对多时相 PSNR 和 SSIM 通常更友好。

---

## 11. 采样策略：必须 deterministic

如果目标是 PSNR，采样时应尽量减少随机性。

推荐：

```text
deterministic Euler sampling
fixed time schedule
no stochastic noise
no random churn
```

原因：

- 随机采样可能提高多样性；
- 但 PSNR 需要固定、准确、像素级一致的输出；
- 随机扰动通常会增加 MSE。

---

## 12. 推荐最终配置

### Path

```text
x_t = (1 - t) * x_cloudy + t * x_clear
```

### Prediction

```text
vθ(t, x_t, c)
```

### Output

```text
x_pred = x_t + (1 - t) * vθ(t, x_t, c)
```

### Loss

```text
L_total =
1.0 * MSE(x_pred, x_clear)
+ 0.2 * MSE((1 - M) * x_pred, (1 - M) * x_cloudy)
+ 0.3 * MSE((1 + M) * (x_pred - x_clear))
+ 0.1 * MSE(vθ, x_clear - x_cloudy)
+ 0.05 * L_velocity_consistency
```

### Sampling

```text
K = 2 或 K = 4
deterministic Euler
no stochastic noise
```

---

## 13. 推荐实验顺序

| 阶段 | 改动 | 目标 |
|---|---|---|
| Exp-1 | cloudy → clear path | 保证 CFM 是 restoration flow |
| Exp-2 | 最终输出 MSE 为主损失 | 直接优化 PSNR |
| Exp-3 | 加 Non-cloud Identity Loss | 防止非云区被改坏 |
| Exp-4 | 加 Cloud-weighted MSE | 提升云区恢复 |
| Exp-5 | 1-step 改为 2-step / 4-step | 提升厚云区域 PSNR |
| Exp-6 | 加残差预测 | 降低学习难度，减少过度修改 |
| Exp-7 | 加 EMRDM teacher distillation | 稳定训练，逼近强 teacher |
| Exp-8 | 加 Mask-guided temporal fusion | 多时相进一步涨分 |

---

## 14. 优先级建议

### 第一优先级：最稳、最容易涨 PSNR

```text
1. 最终输出 MSE
2. Non-cloud Identity Loss
3. Cloud-weighted MSE
4. Residual prediction
```

### 第二优先级：适合进一步提升

```text
1. Multi-segment CFM
2. EMRDM teacher distillation
3. Mask-guided temporal fusion
```

### 第三优先级：可以作为论文创新点包装

```text
1. Cloud-conditioned restoration flow
2. Cloud-aware PSNR-oriented loss
3. Reliability-guided temporal fusion
4. Consistency flow distillation for fast cloud removal
```

---

## 15. 论文贡献点可以这样写

如果要写成论文或课程项目，可以包装为：

> A PSNR-Oriented Conditional Consistency Flow Matching Framework for Remote Sensing Cloud Removal

主要贡献：

1. 提出 cloudy-to-clear conditional consistency flow，将 CFM 从生成任务改造为遥感图像恢复任务；
2. 设计 cloud-aware PSNR-oriented loss，同时增强云区恢复并约束非云区保持；
3. 引入 residual velocity prediction，降低模型学习难度并减少非云区域失真；
4. 采用 multi-segment deterministic sampling，在速度和 PSNR 之间取得更好平衡；
5. 可选地使用 EMRDM teacher distillation 和 mask-guided temporal fusion 进一步提升性能。

---

## 16. 一句话总结

如果已经换成 CFM，提升 PSNR 的关键不是让模型“更会生成”，而是让它成为一个：

> cloud-conditioned residual regression flow

最重要的做法是：

```text
cloudy → clear path
+ final MSE
+ non-cloud identity loss
+ cloud-weighted loss
+ residual prediction
+ 2-step / 4-step deterministic sampling
```
