#!/usr/bin/env python
"""
按 task 拆分 LeRobot 数据集，每个 task 创建一个独立的新数据集。
支持多进程并行处理。
"""

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# ========== 配置 ==========
SOURCE_REPO_ID = "single_arm_v6"
SOURCE_ROOT = Path("/inspire/hdd/project/robot-decision/public/datasets/single_arm_v6")
OUTPUT_ROOT = Path("/inspire/hdd/project/robot-decision/public/datasets/real_aloha1/")  # 新数据集存放目录
MAX_WORKERS = 5  # 并行进程数，根据 CPU 和内存调整

# 需要拆分的 task 列表
TASKS_TO_SPLIT = [
    # "place object in the plate",
    "place the white mug on the plate",
    "place block on the plate",
    "place the red marker in the pen holder",
    "place banana on the plate",
    "stack the right block on the left block",
]


def get_episodes_by_task(meta: LeRobotDatasetMetadata) -> dict[str, list[int]]:
    """获取每个 task 对应的 episode 索引列表"""
    episodes_df = meta.episodes.to_pandas()
    task_to_episodes: dict[str, list[int]] = {}

    for _, row in episodes_df.iterrows():
        ep_idx = row["episode_index"]
        tasks = row["tasks"]  # list of task strings
        for task in tasks:
            if task not in task_to_episodes:
                task_to_episodes[task] = []
            task_to_episodes[task].append(ep_idx)

    return task_to_episodes


def task_to_repo_name(task: str) -> str:
    """将 task 名称转换为合法的 repo 名称"""
    # 替换空格和特殊字符
    name = task.lower().replace(" ", "_").replace("'", "")
    return f"{name}"


def split_dataset_for_task(
    task: str,
    episode_indices: list[int],
    output_root: Path,
    source_repo_id: str,
    source_root: Path,
) -> dict:
    """为单个 task 创建新数据集（独立进程中运行）"""
    repo_name = task_to_repo_name(task)
    new_root = output_root / repo_name

    # 如果目标目录已存在，先删除
    if new_root.exists():
        print(f"  [{task}] 目标目录已存在，删除: {new_root}")
        shutil.rmtree(new_root)

    print(f"  [{task}] 创建新数据集: {repo_name}")
    print(f"  [{task}] 包含 {len(episode_indices)} 个 episodes")

    # 在子进程中加载源数据集
    source_ds = LeRobotDataset(
        repo_id=source_repo_id,
        root=source_root,
        download_videos=False,
    )

    # 获取源数据集的 features（排除默认字段，它们会自动添加）
    source_features = {
        k: v for k, v in source_ds.meta.features.items()
        if k not in ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    }

    # 创建新数据集
    new_ds = LeRobotDataset.create(
        repo_id=repo_name,
        fps=source_ds.fps,
        features=source_features,
        root=new_root,
        robot_type=source_ds.meta.robot_type,
        use_videos=len(source_ds.meta.video_keys) > 0,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    # 加载源数据集中指定的 episodes
    source_subset = LeRobotDataset(
        repo_id=source_repo_id,
        root=source_root,
        episodes=episode_indices,
        download_videos=False,
    )

    # 按 episode 复制数据
    for new_ep_idx, old_ep_idx in enumerate(sorted(episode_indices)):
        print(f"    [{task}] 处理 episode {old_ep_idx} -> {new_ep_idx}")

        # 获取该 episode 的所有帧
        ep_meta = source_ds.meta.episodes[old_ep_idx]
        start_idx = ep_meta["dataset_from_index"]
        end_idx = ep_meta["dataset_to_index"]

        # 计算在 subset 中的相对索引
        # source_subset 只包含 episode_indices 中的数据
        subset_start = source_subset._absolute_to_relative_idx[start_idx]
        subset_end = source_subset._absolute_to_relative_idx[end_idx - 1] + 1

        for rel_idx in range(subset_start, subset_end):
            frame_data = source_subset[rel_idx]

            # 构建 frame dict
            frame = {"task": task}
            for key in source_features:
                if key in frame_data:
                    val = frame_data[key]
                    # 图像数据从 CHW 转回 HWC 格式（add_frame 需要 HWC）
                    if source_features[key]["dtype"] in ["image", "video"]:
                        if isinstance(val, torch.Tensor) and val.ndim == 3:
                            val = val.permute(1, 2, 0)  # CHW -> HWC
                    frame[key] = val

            new_ds.add_frame(frame)

        # 保存 episode
        new_ds.save_episode()

    # 完成并关闭写入器
    new_ds.finalize()
    result = {
        "task": task,
        "repo_name": repo_name,
        "root": str(new_root),
        "total_frames": new_ds.meta.total_frames,
        "total_episodes": new_ds.meta.total_episodes,
    }
    print(f"  [{task}] 完成: {new_root}")
    print(f"  [{task}] 总帧数: {new_ds.meta.total_frames}, 总 episodes: {new_ds.meta.total_episodes}")
    return result


def main():
    print("=" * 60)
    print("按 Task 拆分 LeRobot 数据集")
    print("=" * 60)

    # 加载源数据集元数据
    print(f"\n加载源数据集: {SOURCE_ROOT}")
    source_meta = LeRobotDatasetMetadata(SOURCE_REPO_ID, root=SOURCE_ROOT)

    # 获取每个 task 的 episodes
    task_to_episodes = get_episodes_by_task(source_meta)

    # 显示统计信息
    print("\n源数据集 Task 统计:")
    for task in TASKS_TO_SPLIT:
        if task in task_to_episodes:
            print(f"  - {task}: {len(task_to_episodes[task])} episodes")
        else:
            print(f"  - {task}: 未找到!")

    # 收集需要处理的任务
    tasks_to_process = [
        (task, task_to_episodes[task])
        for task in TASKS_TO_SPLIT
        if task in task_to_episodes
    ]

    if not tasks_to_process:
        print("\n没有找到任何需要处理的 task!")
        return

    # 使用多进程并行处理
    print(f"\n使用 {MAX_WORKERS} 个进程并行处理 {len(tasks_to_process)} 个 tasks...")
    print("=" * 60)

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(
                split_dataset_for_task,
                task,
                episode_indices,
                OUTPUT_ROOT,
                SOURCE_REPO_ID,
                SOURCE_ROOT,
            ): task
            for task, episode_indices in tasks_to_process
        }

        # 收集结果
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n✓ Task 完成: {task}")
            except Exception as e:
                print(f"\n✗ Task 失败: {task}")
                print(f"  错误: {e}")

    # 打印汇总
    print("\n" + "=" * 60)
    print("处理结果汇总:")
    print("=" * 60)
    for r in results:
        print(f"  - {r['task']}: {r['total_episodes']} episodes, {r['total_frames']} frames")
        print(f"    路径: {r['root']}")

    print("\n" + "=" * 60)
    print("所有 Task 拆分完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
