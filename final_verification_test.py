#!/usr/bin/env python3
"""
最终验证测试：确保所有4个评估修复都正确生效
"""

import sys
sys.path.insert(0, 'src')

print("=" * 80)
print("最终验证测试：CARP 评估修复")
print("=" * 80)

# ============================================================================
# 测试 1: ProcessorStep 注册 (问题2的修复)
# ============================================================================
print("\n[测试 1/4] 验证 ProcessorStep 注册...")
try:
    from lerobot.policies.carp import CARPConfig
    from lerobot.processor import ProcessorStepRegistry

    all_steps = ProcessorStepRegistry.list()
    carp_steps = [s for s in all_steps if 'carp' in s]

    expected_carp_steps = ['carp_task_id', 'carp_sample_action_sequence', 'carp_add_temporal_dimension']

    print(f"  总注册步骤数: {len(all_steps)}")
    print(f"  CARP 步骤: {carp_steps}")

    for step_name in expected_carp_steps:
        if step_name in carp_steps:
            print(f"  ✓ {step_name} 已注册")
        else:
            print(f"  ✗ {step_name} 未注册")
            raise AssertionError(f"ProcessorStep '{step_name}' not registered")

    print("  ✅ 测试1通过: 所有 CARP processor steps 已正确注册")

except Exception as e:
    print(f"  ❌ 测试1失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试 2: Task Text → Task Index 映射 (问题1的修复)
# ============================================================================
print("\n[测试 2/4] 验证 Task Text 映射...")
try:
    from lerobot.envs.libero_task_mapping import get_task_index_from_description, LIBERO_TASK_TEXT_TO_INDEX

    # 测试几个已知的任务
    test_cases = [
        ("pick up the alphabet soup and place it in the basket", 24),  # libero_object task 0
        ("pick up the black bowl between the plate and the ramekin and place it on the plate", 34),  # libero_spatial task 0
        ("open the middle drawer of the cabinet", 19),  # libero_goal task 0
        ("put both the alphabet soup and the tomato sauce in the basket", 5),  # libero_10 task 0
    ]

    print(f"  映射字典包含 {len(LIBERO_TASK_TEXT_TO_INDEX)} 个任务")

    for task_desc, expected_idx in test_cases:
        actual_idx = get_task_index_from_description(task_desc)
        if actual_idx == expected_idx:
            print(f"  ✓ '{task_desc[:50]}...' → {actual_idx}")
        else:
            print(f"  ✗ '{task_desc[:50]}...' → {actual_idx} (期望 {expected_idx})")
            raise AssertionError(f"Task mapping failed for: {task_desc}")

    print("  ✅ 测试2通过: Task text → task_index 映射正确")

except Exception as e:
    print(f"  ❌ 测试2失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试 3: RenameObservationsProcessor 包含 (问题3的修复)
# ============================================================================
print("\n[测试 3/4] 验证 RenameObservationsProcessor 包含...")
try:
    from lerobot.policies.carp.configuration_carp import CARPConfig
    from lerobot.policies.carp.processor_carp import make_carp_pre_post_processors
    import torch

    # 创建配置
    config = CARPConfig()
    config.input_features = {
        "observation.images.top": type('obj', (object,), {'type': 'VISUAL', 'shape': (3, 224, 224)})(),
        "observation.state": type('obj', (object,), {'type': 'STATE', 'shape': (14,)})(),
    }
    config.output_features = {
        "action": type('obj', (object,), {'type': 'ACTION', 'shape': (7,)})(),
    }

    # 创建 processor (不需要 dataset_stats，只检查结构)
    preprocessor, postprocessor = make_carp_pre_post_processors(
        config=config,
        dataset_stats=None,
    )

    # 检查 preprocessor 的 steps
    step_names = []
    for step in preprocessor.steps:
        step_class_name = step.__class__.__name__
        step_names.append(step_class_name)

    print(f"  Preprocessor 包含 {len(step_names)} 个步骤:")
    for name in step_names:
        print(f"    - {name}")

    if "RenameObservationsProcessorStep" in step_names:
        print("  ✓ RenameObservationsProcessorStep 已包含")
    else:
        print("  ✗ RenameObservationsProcessorStep 未包含")
        raise AssertionError("RenameObservationsProcessorStep not included in preprocessor")

    print("  ✅ 测试3通过: RenameObservationsProcessor 已正确包含")

except Exception as e:
    print(f"  ❌ 测试3失败: {e}")
    sys.exit(1)

# ============================================================================
# 测试 4: Task ID 类型转换 (问题4的修复)
# ============================================================================
print("\n[测试 4/4] 验证 Task ID 类型转换...")
try:
    import numpy as np
    import torch
    from lerobot.envs.utils import preprocess_observation

    # 模拟环境观测 (包含 task_id)
    mock_observation = {
        "pixels": {
            "top": np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8),
        },
        "robot_state": {
            "joint": {
                "pos": np.random.randn(1, 7).astype(np.float32),
            },
        },
        "task_id": np.array([15], dtype=np.int64),  # numpy array
    }

    print(f"  输入 task_id 类型: {type(mock_observation['task_id'])}")
    print(f"  输入 task_id 值: {mock_observation['task_id']}")

    # 预处理
    processed = preprocess_observation(mock_observation)

    if "task_id" not in processed:
        print("  ✗ task_id 字段丢失")
        raise AssertionError("task_id field missing after preprocessing")

    task_id = processed["task_id"]
    print(f"  输出 task_id 类型: {type(task_id)}")
    print(f"  输出 task_id dtype: {task_id.dtype}")
    print(f"  输出 task_id 值: {task_id}")

    # 验证类型
    if not isinstance(task_id, torch.Tensor):
        print("  ✗ task_id 不是 torch.Tensor")
        raise AssertionError(f"Expected torch.Tensor, got {type(task_id)}")

    if task_id.dtype not in (torch.long, torch.int64):
        print("  ✗ task_id dtype 不是 torch.long")
        raise AssertionError(f"Expected torch.long, got {task_id.dtype}")

    print("  ✓ task_id 类型正确: torch.Tensor")
    print("  ✓ task_id dtype 正确: torch.int64")

    # 测试 nn.Embedding 调用
    task_embed = torch.nn.Embedding(40, 128)
    try:
        embedded = task_embed(task_id)
        print(f"  ✓ nn.Embedding 调用成功, 输出 shape: {embedded.shape}")
    except Exception as e:
        print(f"  ✗ nn.Embedding 调用失败: {e}")
        raise

    print("  ✅ 测试4通过: Task ID 类型转换正确")

except Exception as e:
    print(f"  ❌ 测试4失败: {e}")
    sys.exit(1)

# ============================================================================
# 最终总结
# ============================================================================
print("\n" + "=" * 80)
print("✅ 所有4个修复验证通过!")
print("=" * 80)
print("\n修复总结:")
print("  1. ✓ Task ID 映射 (libero_task_mapping.py)")
print("  2. ✓ ProcessorStep 注册 (policies/carp/__init__.py)")
print("  3. ✓ RenameObservationsProcessor 包含 (processor_carp.py)")
print("  4. ✓ Task ID 类型转换 (envs/utils.py)")
print("\n现在可以使用以下命令进行训练和评估:")
print("\n训练:")
print("  lerobot-train --policy=carp --dataset.repo_id=/path/to/libero \\")
print("    --policy.ar_training_mode=false --batch_size=256 --steps=200000")
print("\n评估:")
print("  lerobot-eval --policy.path=/path/to/checkpoint \\")
print("    --env.type=libero --env.task=libero_spatial --eval.n_episodes=50")
print("\n" + "=" * 80)
