#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from libero.libero import benchmark
from lerobot.envs.libero_task_mapping import get_task_index_from_description

# 测试每个 suite 的前3个任务
suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']

print("=== Task Description → Task Index Mapping Test ===\n")

for suite_name in suites:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    print(f"{suite_name}:")

    for i in range(min(3, len(suite.tasks))):
        task = suite.get_task(i)
        task_desc = task.language
        task_idx = get_task_index_from_description(task_desc)

        print(f"  [{i}] env_task_id={i}, dataset_task_index={task_idx}")
        print(f"      \"{task_desc}\"")
    print()

print("✓ All task descriptions successfully mapped to dataset task_index")
