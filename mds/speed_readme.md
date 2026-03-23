# Policy Inference Speed Benchmark

这个脚本用于测试lerobot框架下某个policy推理一个action chunk的时间和推理频率。

## 功能特性

- 测试完整的推理流程（预处理 + 推理 + 后处理）
- 分别统计各阶段的时间
- 支持GPU预热，避免首次推理的初始化延迟
- 计算推理频率（Hz）
- 提供详细的统计信息（均值、标准差、最小值、最大值）

## 使用方法

### 基本用法

```bash
python testspeed2.py --policy_path outputs/pi05_0121/checkpoints/005000
```

### 完整参数说明

```bash
python testspeed2.py \
    --policy_path outputs/pi05_0121/checkpoints/005000 \
    --device cuda:0 \
    --batch_size 1 \
    --n_warmup 10 \
    --n_iterations 100
```

### 参数说明

- `--policy_path`: Policy checkpoint路径（必需）
- `--device`: 推理设备，默认为 `cuda:0`，也可以使用 `cuda:1`, `cpu` 等
- `--batch_size`: Batch大小，默认为 `1`
- `--n_warmup`: GPU预热迭代次数，默认为 `10`
- `--n_iterations`: 测试迭代次数，默认为 `100`（越多越准确，但耗时更长）

## 输出示例

```
使用设备: cuda:0
正在加载policy: outputs/pi05_0121/checkpoints/005000
Policy加载成功
Policy类型: PI05Policy
Action chunk size: 16
正在创建预处理器和后处理器...
预处理器和后处理器创建成功
正在创建模拟观测数据 (batch_size=1)...
模拟观测数据创建成功
输入特征: ['task', 'observation.state', 'observation.images.front', 'observation.images.wrist']
正在进行 10 次预热...
预热完成

开始测试推理速度 (迭代次数: 100)...
完成 10/100 次迭代
完成 20/100 次迭代
...
完成 100/100 次迭代

============================================================
推理速度测试结果
============================================================

Action Chunk Size: 16

预处理时间:
  平均: 2.35 ms
  标准差: 0.18 ms
  最小: 2.10 ms
  最大: 3.20 ms

推理时间 (predict_action_chunk):
  平均: 45.67 ms
  标准差: 1.23 ms
  最小: 44.20 ms
  最大: 52.10 ms

后处理时间:
  平均: 0.82 ms
  标准差: 0.05 ms
  最小: 0.75 ms
  最大: 1.10 ms

总时间 (端到端):
  平均: 48.84 ms
  标准差: 1.28 ms
  最小: 47.50 ms
  最大: 55.40 ms

推理频率: 20.47 Hz
============================================================

测试完成!
```

## 支持的Policy类型

脚本当前已支持：
- PI05Policy
- PI0FastPolicy

可以通过修改导入和加载代码来支持其他policy类型。

## 注意事项

1. **GPU预热**：首次推理时GPU需要初始化，会比后续推理慢很多。脚本会先进行预热迭代。

2. **同步操作**：在使用CUDA时，脚本会在测量时间前后调用 `torch.cuda.synchronize()` 确保GPU操作完成。

3. **内存占用**：如果遇到OOM（内存不足）错误，可以减小 `batch_size`。

4. **测试结果**：测试使用随机生成的模拟数据，仅用于速度测试，不代表实际推理效果。

## 测试不同Policy

### 测试PI0Fast

```python
# 修改 testspeed2.py 中的policy加载部分：
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

# 在 main() 函数中：
policy = PI0FastPolicy.from_pretrained(args.policy_path).to(device).eval()
```

### 测试其他Policy

类似地导入和加载其他policy类型即可。

## 扩展功能

你可以根据需要添加以下功能：
- 测试不同batch size的性能
- 测试使用不同精度（FP32, FP16, BF16）的性能
- 保存测试结果到文件
- 生成性能对比图表

## 常见问题

**Q: 为什么我的推理频率很低？**  
A: 可能的原因：
- GPU性能不足
- Policy模型较大
- Action chunk size较大
- 需要检查GPU利用率

**Q: 如何提高推理速度？**  
A: 可以尝试：
- 使用混合精度（FP16/BF16）
- 使用更强的GPU
- 减小模型大小或action chunk size
- 使用TensorRT等加速工具

**Q: 测试结果不稳定怎么办？**  
A: 增加 `n_iterations` 参数，例如设置为 `200` 或 `500`，会得到更稳定的统计结果。
