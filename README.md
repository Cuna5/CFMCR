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

### Test-Time Augmentation（TTA）

TTA 通过对输入做 4 种几何变换（原图、水平翻转、垂直翻转、180° 旋转），分别推理后逆变换取平均，可获得 +0.1~0.3 dB PSNR 提升，推理时间变为 4 倍，不需要重新训练。

开启 TTA：

```bash
python main.py --base configs/example_training/cuhk_cfm.yaml \
    -t false \
    model.params.sampler_config.params.tta=True
```

TTA 可与多步采样叠加使用（推理时间为 4×num_steps）。建议仅在 test / predict 时开启，训练 validation 保持关闭。

## 画质指标相关 YAML 参数

下面以 `configs/example_training/cuhk_cfm.yaml` 和 `configs/example_training/cuhkv2_cfm.yaml` 为准。表中的推荐范围是 PSNR / SSIM / RMSE 调参起点，不是固定最优值；建议一次只改 1-2 个核心参数，并用同一 checkpoint / 同一测试集做对照。

### 推理阶段参数

这些参数不需要重新训练，最适合先做快速评估。

| 参数路径 | 当前默认值 | 推荐参考值 | 对画质指标的影响 |
|----------|------------|------------|------------------|
| `model.params.sampler_config.params.num_steps` | `1` | 快速：`1`；质量折中：`2~4`；实验：`4~8` | 推理步数越多，通常 PSNR / SSIM 更稳，但耗时近似线性增加。`1` 是真正一步 CFM。 |
| `model.params.sampler_config.params.tta` | `False` | test / predict 可设 `True` | 4 种几何变换平均，通常可提升约 `0.1~0.3 dB` PSNR，耗时变为 `4×num_steps`。 |
| `model.params.sampler_config.params.s_churn` | `0.0` | 稳定基线：`0.0`；多步实验：`0.05~0.5` | 仅 `num_steps > 1` 生效。给多步 Euler 路径加随机扰动，可能改善细节，也可能损伤 PSNR，需扫参验证。 |
| `model.params.sampler_config.params.s_tmin` | `0.0` | `0.0~0.1` | 只在 `sigma >= s_tmin` 的步骤加扰动。提高到 `0.05/0.1` 可避免末端低噪声阶段继续注入噪声。 |
| `model.params.sampler_config.params.s_tmax` | `100000000.0` | 全程：`100000000.0`；只扰动起点段：`0.5~1.0` | 只在 `sigma <= s_tmax` 的步骤加扰动。CFM 当前 `sigma_max=1.0`，大值等价于不设上限。 |
| `model.params.sampler_config.params.s_noise` | `1.0` | `0.8~1.1` | 随机扰动噪声幅度。过高容易引入伪影，过低则 `s_churn` 效果不明显。 |
| `model.params.sampler_config.params.discretization_config.params.sigma_min` | `0.001` | `0.0005~0.005`，优先保持和训练一致 | 多步推理终点附近的最小 sigma。过大可能残留偏差，过小收益通常有限。 |
| `model.params.sampler_config.params.discretization_config.params.sigma_max` | `1.0` | 固定 `1.0` | CFM 一步起点，表示从含云图像 `μ` 出发。不建议改，否则和训练路径不一致。 |
| `model.params.sampler_config.params.discretization_config.params.rho` | `1` | 优先固定 `1`；实验可试 `1~3` | 控制多步 sigma 时间表形状。训练使用线性 sigma，推理也建议保持一致。 |

### 损失函数和训练目标

这些参数需要重新训练或至少继续微调，直接决定模型学到的端点质量。

| 参数路径 | 当前默认值 | 推荐参考值 | 对画质指标的影响 |
|----------|------------|------------|------------------|
| `model.params.loss_fn_config.params.loss_type` | `"charbonnier"` | `"charbonnier"`；对照可试 `"l1"` / `"l2"` | Charbonnier 通常比 L2 更不容易过平滑，是当前推荐默认项。 |
| `model.params.loss_fn_config.params.charbonnier_eps` | `1.0e-3` | `1.0e-4~1.0e-2` | 越小越接近 L1，细节可能更锐；越大越平滑，训练可能更稳。 |
| `model.params.loss_fn_config.params.clean_endpoint_loss_weight` | `1.0` | `1.0~3.0` | 直接监督一步输出靠近 `x_clean`，通常是最直接服务 PSNR / SSIM / RMSE 的权重。过高可能削弱多步一致性。 |
| `model.params.loss_fn_config.params.velocity_anchor_loss_weight` | `1.0` | `0.5~2.0` | 约束速度方向 `vθ ≈ x_clean - μ`。过低容易方向漂移，过高可能限制 teacher consistency 的收益。 |
| `model.params.loss_fn_config.params.endpoint_loss_weight` | `0.5` | `0.1~1.0` | EMA teacher 端点一致性。teacher 尚不稳定时过高会拖累端点指标。 |
| `model.params.loss_fn_config.params.consistency_loss_weight` | `0.5` | `0.1~1.0` | EMA teacher 速度一致性。通常和 `endpoint_loss_weight` 联动调整。 |
| `model.params.loss_fn_config.params.start_pair_prob` | `0.35` | `0.25~0.6` | 提高 `sigma≈1` 起点段采样概率，强化一步推理。过高可能降低中后段路径覆盖。 |
| `model.params.loss_fn_config.params.consistency_warmup_steps` | `2000` | `1000~5000`；训练很长可试 `10000` | 前期让真实监督先主导，避免随机 teacher 过早影响训练。训练不稳时增大。 |
| `model.params.loss_fn_config.params.num_steps` | `40` | `40~100` | 训练时采样相邻 sigma pair 的时间表密度。更大覆盖更细，但训练目标更分散、计算略增。 |
| `model.params.loss_fn_config.params.cloud_mask_key` | `"M"` | CUHK 保持 `"M"` | 云区加权使用的 mask。mask 不可靠时不要盲目提高云区权重。 |
| `model.params.loss_fn_config.params.cloud_loss_weight` | `2.0` | `1.0~3.0`；重点云区可试 `4.0` | 云区像素权重。`1.0` 等价关闭；过高可能牺牲非云区色彩一致性。 |
| `model.params.loss_fn_config.params.cloud_weight_velocity_anchor` | `True` | mask 可靠：`True`；mask 噪声大：`False` | 是否同步加权速度锚点。开启更聚焦云区恢复，关闭更保守。 |
| `model.params.loss_fn_config.params.discretization_config.params.sigma_min` | `0.001` | `0.0005~0.005` | 训练路径终点。建议和 sampler 的 `sigma_min` 保持一致。 |
| `model.params.loss_fn_config.params.discretization_config.params.sigma_max` | `1.0` | 固定 `1.0` | 对应 CFM 起点 `t=0`。不建议改。 |
| `model.params.loss_fn_config.params.discretization_config.params.rho` | `1` | 固定 `1`；做消融可试 `1~3` | 当前 CFM 按线性 sigma / 均匀 t 训练，优先保持 `1`。 |

### 优化器、EMA 和训练时长

这些参数主要影响收敛质量和最终 checkpoint 的稳定性。

| 参数路径 | 当前默认值 | 推荐参考值 | 对画质指标的影响 |
|----------|------------|------------|------------------|
| `model.params.ckpt_path` | 默认注释关闭 | 从同一 CFM 任务 checkpoint 继续训练 | warm-start 往往比从零训练更快收敛；跨数据集 checkpoint 需要重新验证，可能带来域偏移。 |
| `model.base_learning_rate` | `1.0e-4` | 稳定：`5.0e-5~1.0e-4`；加速：`1.5e-4~2.0e-4` | 学习率过高会导致指标波动，过低会收敛慢或欠拟合。 |
| `model.params.teacher_ema_decay` | `0.9999` | `0.999~0.99995` | teacher 越慢越稳但越滞后。短训或前期不稳可试 `0.999/0.9995`。 |
| `model.params.scheduler_config.params.warm_up_steps` | `2000` | `1000~5000` | warmup 太短易早期震荡，太长会拖慢收敛。 |
| `model.params.scheduler_config.params.lr_start` | `0.01` | `0.001~0.05` | warmup 起始倍率，乘在 `base_learning_rate` 上。一般不用频繁改。 |
| `model.params.scheduler_config.params.lr_max` | `1.0` | `0.8~1.2` | 峰值学习率倍率。多数实验保持 `1.0`。 |
| `model.params.scheduler_config.params.lr_min` | `0.01` | `0.001~0.05` | 后期最小学习率倍率。微调阶段可更低，减少指标抖动。 |
| `model.params.scheduler_config.params.max_decay_steps` | CUHK-CR1: `100000`；CUHK-CR2: `400000` | 设为接近总训练 step | 太小会过早降学习率导致欠拟合，太大则后期可能不够稳定。 |
| `data.params.batch_size` | `4` | 显存允许下 `2~8`；预测阶段建议 `1` | batch 越大梯度越稳，但可能降低增强随机性。预测保存逐张图时 batch 太大可能不合适。 |
| `lightning.trainer.accumulate_grad_batches` | `1` | `1~4` | 改变等效 batch size。显存小但想稳定训练时可提高。 |
| `lightning.trainer.max_epochs` | CUHK-CR1: `500`；CUHK-CR2: `100` | 跑到验证指标平台期；CUHK-CR2 长训可试 `200~1000+` | 训练不足会欠拟合；训练过久可能过拟合，需看 validation RMSE / PSNR。 |
| `lightning.modelcheckpoint.params.monitor` | `"RMSE"` | 关注 RMSE 时保持 `"RMSE"` | 决定按哪个指标保存最优模型。若主指标换成 PSNR / SSIM，需要确认日志里有对应键。 |

### 指标计算和可视化口径

这些参数不一定改变模型输出，但会影响你看到的指标或图像是否可比较。

| 参数路径 | 当前默认值 | 推荐参考值 | 对画质指标的影响 |
|----------|------------|------------|------------------|
| `model.params.image_metrics` | `"evaluator"` | 保持 `"evaluator"` | 决定评估指标实现。为了不同实验可比，建议固定不变。 |
| `model.params.to_rgb_config.target` | `sgm.util.nir_to_rgb` | CUHK + NIR 保持当前值 | 影响日志图像的 RGB 展示口径。若改通道定义，需要同步修改，否则可视化会误导判断。 |

### 数据和增强

数据配置对画质指标影响很大，尤其是 clean / cloudy / NIR 是否严格配对。

| 参数路径 | 当前默认值 | 推荐参考值 | 对画质指标的影响 |
|----------|------------|------------|------------------|
| `data.params.train.params.datasets_dir` | 数据集路径 | 使用正确 split，避免 train/test 混用 | 路径或配对错误会让指标失真，是最先检查的项。 |
| `data.params.train.params.nir_datasets_dir` | NIR 数据路径 | 与 RGB 数据严格同 split / 同命名配对 | 当前网络 `in_channels=8/out_channels=4` 依赖 NIR 通道；缺失或错配会严重影响指标。 |
| `data.params.train.params.augment` | `True` | 训练：`True`；验证/测试/预测：`False` | 成对增强可提升泛化；评估阶段必须关闭以保证指标可复现。 |
| `data.params.train.params.hflip_p` | `0.5` | `0.3~0.7` | 水平翻转概率。遥感图像通常适合开启。 |
| `data.params.train.params.vflip_p` | `0.5` | `0.3~0.7` | 垂直翻转概率。和水平翻转一起扩大空间分布。 |
| `data.params.train.params.rot90_p` | `0.5` | `0.25~0.75` | 90 度旋转增强概率。过高时需确认数据方向本身不携带特殊先验。 |
| `data.params.validation/test/predict.params.isTrain` | `False` | 固定 `False` | 确保评估不走训练随机逻辑，保证 PSNR / SSIM / RMSE 可复现。 |

### 模型容量参数

这些参数会改变网络容量，通常需要重新训练。除非显存不足或明显欠拟合，否则建议先不要动模型结构。

| 参数路径 | 当前默认值 | 推荐参考值 | 对画质指标的影响 |
|----------|------------|------------|------------------|
| `model.params.network_config.params.in_channels` | `8` | 当前 CUHK + NIR 固定 `8` | 输入通道数，必须和 `x + cond_image` 拼接后的通道一致。改错会直接维度报错或训练无效。 |
| `model.params.network_config.params.out_channels` | `4` | 当前 CUHK + NIR 固定 `4` | 输出 clean 图像通道数，必须和 label 通道一致。 |
| `model.params.network_config.params.patch_size` | `[1, 1]` | 优先固定 `[1, 1]` | 更大 patch 可能省显存但损失像素级细节。 |
| `model.params.network_config.params.widths` | `[128,256,384,768]` | 显存紧张可降到约 `0.75×`；追求质量可试 `1.25~1.5×` | 主干通道宽度，直接影响容量和显存。 |
| `model.params.network_config.params.depths` | `[2,2,2,2]` | `2~4` | 每层 block 数。加深可能提升复杂场景恢复，但训练更慢、过拟合风险更高。 |
| `model.params.network_config.params.d_ffs` | `[256,512,768,1536]` | 约为对应 `widths` 的 `2×~4×` | FFN 宽度，影响纹理和细节建模能力。 |
| `model.params.network_config.params.self_attns[*].type` | 前两层 `neighborhood`，后两层 `global` | 优先保持当前组合 | 局部注意力保细节，全局注意力建模长程结构。随意替换会改变速度和指标。 |
| `model.params.network_config.params.self_attns[*].d_head` | `64` | `32` 或 `64` | attention head 维度。通常保持 `64`。 |
| `model.params.network_config.params.self_attns[*].kernel_size` | `7` | `5/7/9` | neighborhood attention 感受野。更大可能改善大云块，但显存和耗时增加。 |
| `model.params.network_config.params.dropout_rate` | `[0.0,0.0,0.0,0.1]` | `0.0~0.2` | 数据少或过拟合时提高；欠拟合或输出偏平滑时降低。 |
| `model.params.network_config.params.mapping_depth` | `2` | `2~4` | 时间嵌入映射网络深度。一般影响小于主干容量。 |
| `model.params.network_config.params.mapping_width` | `768` | 与最高层 width 接近 | 时间嵌入宽度。通常和主干最高通道保持一致。 |
| `model.params.network_config.params.mapping_d_ff` | `1536` | `2×mapping_width` 左右 | 时间映射 FFN 宽度，通常保持当前比例。 |
| `model.params.network_config.params.mapping_dropout_rate` | `0.1` | `0.0~0.2` | 时间映射 dropout。过高可能损伤精细端点预测。 |
| `model.params.conditioner_config.params.emb_models[0].ucg_rate` | `0.0` | 监督去云保持 `0.0` | 无条件 dropout 比例。当前任务不依赖 CFG，开启可能降低条件图利用率。 |

### 不建议随意修改的 CFM 固定项

这些 YAML 项会改变方法定义或数据契约，不属于常规画质调参项：`model.target`、`model.params.sigma_st_config.target`、`model.params.denoiser_config.params.scaling_config.target`、`model.params.network_wrapper`、`model.params.first_stage_config.target`、`model.params.input_key`、`model.params.mean_key`、`model.params.conditioner_config.params.emb_models[0].input_key`。除非你要换模型范式或数据格式，否则建议保持当前配置。

### 推荐调参顺序

1. 不重训先试：`sampler_config.params.tta=True`，再扫 `sampler_config.params.num_steps=2/3/4`。
2. 多步有效后再试：`s_churn=0.05/0.1/0.2/0.5`，并观察是否提升 PSNR / SSIM、是否引入伪影。
3. 重新训练时优先扫：`clean_endpoint_loss_weight=1.0/1.5/2.0`、`cloud_loss_weight=1.5/2.0/3.0`、`start_pair_prob=0.25/0.35/0.5`。
4. 若训练不稳，再调：`base_learning_rate=5e-5/1e-4/2e-4`、`consistency_warmup_steps=2000/5000`、`teacher_ema_decay=0.999/0.9995/0.9999`。
5. 只有确认不是训练目标问题后，再改模型容量参数，例如 `widths`、`depths`、`dropout_rate`。

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

## RICE1 / RICE2

RICE 数据集使用 RGB 三通道，因此对应配置的网络输入/输出为
`in_channels=6`、`out_channels=3`。当前本地数据按固定 seed 划分：

| 数据集 | 总数 | Train | Validation | Test |
|--------|------|-------|------------|------|
| RICE1 | 500 | 400 | 50 | 50 |
| RICE2 | 736 | 588 | 74 | 74 |

训练 CFM：

```bash
python main.py --enable_tf32 true --base configs/example_training/rice1_cfm.yaml
python main.py --enable_tf32 true --base configs/example_training/rice2_cfm.yaml
```

训练 MeanFlow：

```bash
python main.py --enable_tf32 true --base configs/example_training/rice1_meanflow.yaml
python main.py --enable_tf32 true --base configs/example_training/rice2_meanflow.yaml
```

默认服务器路径为：

```text
/data1/home/ely/Projects/CFMCR/dataset/RICE1
/data1/home/ely/Projects/CFMCR/dataset/RICE2
```

RICE2 使用 `mask/` 中任意非黑像素作为云影响区域；RICE1 没有原生 mask，
使用 cloudy/label 的绝对差生成软 mask。两者都只将 mask 用于损失加权和分区
指标，不作为网络输入。

## 耗时统计

如需比较速度，打开：

```yaml
model:
  params:
    count_sample_time: True
    count_train_time: True
```

`sample_time` 会覆盖 validation / test / predict 的 sample + decode 阶段；`train_time` 和 `train_time_avg` 会记录训练 batch 耗时，不包含 dataloader 取数时间。
