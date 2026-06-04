# 改进方法说明

本文档说明各项改进在 EMRDM-ODE（Consistency Flow Matching 去云框架）中的作用与原理。

---

## 1. Charbonnier Loss（替换 L2）

**作用：** 替换所有监督损失项（`clean_endpoint_loss`、`velocity_anchor_loss`）中的 L2 范数。

**原理：**

$$\mathcal{L}_{char}(r) = \sqrt{r^2 + \varepsilon^2}, \quad \varepsilon = 10^{-3}$$

L2（均方误差）对大残差的惩罚是平方级别的，会驱使模型过度平滑输出以压低少数大误差像素，导致云区恢复结果模糊。L1 在零点不可微，训练不稳定。Charbonnier 是两者的折中：在大残差处近似 L1（线性惩罚，不过度平滑），在小残差处近似 L2（平滑可微，梯度稳定）。

在本项目中，`clean_endpoint_loss` 直接监督 1 步输出 $f_\theta$ 对齐 $x_{clean}$，是 PSNR/SSIM 指标的直接优化目标。用 Charbonnier 替换 L2 后，模型对云区大误差的惩罚更线性，能保留更多高频纹理细节，而不是一味平均。

**配置位置：** `loss_fn_config.params.loss_type: "charbonnier"`，`charbonnier_eps: 1e-3`。

---

## 2. 数据增强（水平/垂直翻转 + 90° 旋转）

**作用：** 在训练时对 `(含云图, 无云图, 云掩码)` 三元组同步施加随机几何变换，扩充有效训练样本。

**原理：**

卫星遥感图像具有旋转/翻转不变性——同一地物从不同方向拍摄，语义不变。通过随机水平翻转、垂直翻转和 90° 旋转，单张样本可以产生最多 8 种等价视图，相当于将数据集扩大 8 倍。

对 CFM 训练的具体意义：
- 模型的速度场 $v_\theta(x_t, t, \mu)$ 需要学习从含云图到无云图的映射，几何增强迫使模型学习对方向不敏感的特征，减少对训练集中特定地物方向的过拟合。
- 云的形态本身也是各向同性的，翻转/旋转后云区掩码同步变换，不会引入标签噪声。

**注意：** 增强必须对 `input`（$x_{clean}$）、`mu`（含云图）、`M`（云掩码）三者同步施加相同变换，否则掩码与图像错位会污染云区加权 loss。

---

## 3. 云区加权 Loss

**作用：** 对云覆盖像素施加更高的损失权重，使模型在云区的恢复质量更优先。

**原理：**

标准 MSE/Charbonnier 对所有像素等权，而云区像素才是真正需要恢复的区域，非云区像素本身已经是正确值。若等权训练，模型会把大量梯度浪费在已经准确的非云区像素上。

本项目的实现（`loss_cfm.py`）：从 batch 中读取云掩码 `M`（1=云，0=非云），构造权重图：

$$w_i = 1 + (\lambda_{cloud} - 1) \cdot M_i, \quad \lambda_{cloud} = 2.0$$

然后按样本均值归一化，使整体 loss 尺度不变：

$$\tilde{w}_i = \frac{w_i}{\text{mean}(w_i)}$$

加权后 `clean_endpoint_loss` 和（可选的）`velocity_anchor_loss` 对云区像素的梯度贡献翻倍，模型会更努力地恢复云下地物，而不是靠非云区的大量像素"稀释"云区误差。

**配置：** `cloud_loss_weight: 2.0`，`cloud_weight_velocity_anchor: true`。

---

## 4. Test-Time Augmentation（TTA）推理

**作用：** 推理时对同一输入施加多种几何变换，分别推理后反变换取平均，提升单张图像的预测稳定性。

**原理：**

神经网络对输入的微小几何变化不是完全等变的，同一场景翻转后推理的结果与直接推理结果存在细微差异。TTA 利用这一点：

1. 对含云图 $\mu$ 施加 $K$ 种变换（翻转/旋转），得到 $\{\mu_k\}$。
2. 对每个 $\mu_k$ 独立做 1 步 CFM 推理，得到 $\{\hat{x}_k\}$。
3. 将每个 $\hat{x}_k$ 反变换回原始方向，取像素均值：$\hat{x} = \frac{1}{K}\sum_k T_k^{-1}(\hat{x}_k)$。

平均操作相当于对模型预测的随机误差做集成，能有效降低单次推理的方差，在 PSNR/SSIM 上通常有 0.1–0.3 dB 的稳定提升，且无需重新训练。

代价是推理时间增加 $K$ 倍（$K=8$ 时为 8 倍）。对于离线评测场景，这是无成本的提升。

**配置：** `sampler_config.params.tta: true`（见 `Enhance.md` 中的 TTA 配置段）。

---

## 5. EMRDM 式随机扰动

**作用：** 在 CFM 的 OT 直线路径上加入小幅随机噪声扰动，模拟原始 EMRDM 扩散过程的随机性，改善模型对分布外输入的鲁棒性。

**原理：**

标准 CFM 训练点是确定性的 OT 插值：

$$x_{t_n} = (1-t_n)\mu + t_n x_{clean}$$

没有任何随机性。原始 EMRDM 是扩散模型，训练时每个时间步都会加入高斯噪声，模型因此学会了处理带噪输入。

EMRDM 式扰动在 CFM 路径点上叠加小幅噪声：

$$\tilde{x}_{t_n} = x_{t_n} + \delta \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\delta$ 是一个小的扰动幅度（如 $0.05\sigma_n$）。这样做的效果：
- 训练时模型见过带噪的中间状态，推理时若输入含云图本身有传感器噪声，模型不会过拟合到干净的 OT 路径。
- 轻微扰动相当于隐式数据增强，可以缓解 CFM 在厚云区域的过平滑问题。
- 扰动幅度需要控制，过大会破坏 OT 路径的直线性，导致 teacher 一致性目标失效。

---

## 6. MS-SSIM Endpoint Loss

**作用：** 在 `clean_endpoint_loss` 之外，额外对 1 步输出 $f_\theta$ 施加多尺度结构相似性损失，改善视觉质量和 SSIM 指标。

**原理：**

MSE/Charbonnier 是像素级损失，对亮度误差敏感，但对结构、纹理、对比度不敏感。MS-SSIM 在多个下采样尺度上计算亮度、对比度、结构三项相似度的乘积：

$$\mathcal{L}_{MS\text{-}SSIM} = 1 - \prod_{s=1}^{S} \text{SSIM}_s(f_\theta, x_{clean})^{\beta_s}$$

多尺度设计使其同时对大范围结构（低频）和局部纹理（高频）敏感。

在本项目中（`loss_cfm.py` 的 `_ms_ssim_loss` 方法），实现了 3 尺度的 MS-SSIM，使用 Gaussian 窗口计算局部统计量。与 Charbonnier 联合使用时，Charbonnier 保证像素精度（PSNR/RMSE），MS-SSIM 保证结构保真度（SSIM），两者互补。

推荐权重范围：`ssim_endpoint_loss_weight: 0.05–0.2`，过高会与 Charbonnier 产生梯度冲突，导致 PSNR 下降。

**配置：** `loss_fn_config.params.ssim_endpoint_loss_weight: 0.1`。

---

## 7. Non-cloud Identity Loss

**作用：** 对非云区像素施加恒等约束，惩罚模型对已清晰像素的修改，直接提升全图 PSNR。

**原理：**

CFM 的端点映射 $f_\theta$ 会对整张图像做预测，包括非云区。理想情况下，非云区的输出应等于含云图输入（因为那里本来就是清晰的）。但模型在训练时没有显式约束，可能会对非云区引入细微的亮度偏移或纹理改变，拉低全图 PSNR。

Non-cloud Identity Loss 显式惩罚这种偏移：

$$\mathcal{L}_{id} = \|(1-M) \odot (f_\theta - \mu)\|^2$$

其中 $(1-M)$ 是非云区掩码。这个 loss 要求模型在非云区的输出尽量等于输入含云图（即恒等映射）。

效果：
- 非云区 PSNR 直接提升，因为模型不再"乱动"已经正确的像素。
- 间接帮助云区：模型的容量更集中于云区恢复，而不是同时调整非云区。
- 与云区加权 loss 形成互补：云区加权 loss 推动云区恢复，Identity Loss 约束非云区不变。

**配置：** `loss_fn_config.params.non_cloud_identity_loss_weight: 0.1–0.5`，建议从小值开始，避免过强约束限制云区恢复能力。
