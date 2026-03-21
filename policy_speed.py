"""
Enhanced Policy Inference Speed Test Script
增强版的推理速度测试脚本，支持多种policy类型和更多功能
"""
import os
import sys
import time
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors


# 支持的policy类型
POLICY_CLASSES = {
    "pi05": PI05Policy,
    "pi0_fast": PI0FastPolicy,
    "pi0": PI0Policy,
}


def detect_policy_type(policy_path: str) -> str:
    """
    自动检测policy类型
    
    Args:
        policy_path: Policy checkpoint路径
        
    Returns:
        policy类型字符串
    """
    # 尝试从路径名称推断
    path_lower = policy_path.lower()
    if "pi05" in path_lower or "pi0.5" in path_lower:
        return "pi05"
    elif "pi0_fast" in path_lower:
        return "pi0_fast"
    elif "pi0" in path_lower:
        return "pi0"
    
    # 默认返回pi05
    return "pi05"


def create_dummy_observation(policy, device, batch_size=1):
    """
    创建一个模拟的观测数据，用于测试推理速度
    
    Args:
        policy: 加载的policy模型
        device: 推理设备
        batch_size: batch大小
        
    Returns:
        包含所有必要输入的字典
    """
    cfg = policy.config
    
    # 创建模拟观测数据
    obs_dict = {}
    
    # 添加任务描述
    obs_dict["task"] = ["Pick up the object"] * batch_size
    
    # 添加state
    if "observation.state" in cfg.input_features:
        state_dim = cfg.input_features["observation.state"].shape[0]
        obs_dict["observation.state"] = torch.randn(batch_size, state_dim, device=device)
    
    # 添加图像
    for img_key in cfg.image_features:
        # 默认使用224x224的图像
        img_size = 224  # 可以从config中读取
        obs_dict[img_key] = torch.rand(batch_size, 3, img_size, img_size, device=device)
    
    return obs_dict


def warmup_inference(policy, preprocessor, obs_dict, n_warmup=10):
    """
    预热GPU，避免首次推理时的初始化延迟
    
    Args:
        policy: 加载的policy模型
        preprocessor: 预处理器
        obs_dict: 输入观测数据
        n_warmup: 预热次数
    """
    print(f"正在进行 {n_warmup} 次预热...")
    for _ in range(n_warmup):
        obs_processed = preprocessor(obs_dict)
        with torch.inference_mode():
            _ = policy.predict_action_chunk(obs_processed)
    
    # 同步CUDA以确保所有操作完成
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print("预热完成")


def benchmark_inference(policy, preprocessor, postprocessor, obs_dict, n_iterations=100, verbose=True):
    """
    测试推理速度
    
    Args:
        policy: 加载的policy模型
        preprocessor: 预处理器
        postprocessor: 后处理器
        obs_dict: 输入观测数据
        n_iterations: 测试迭代次数
        verbose: 是否打印进度
        
    Returns:
        包含统计信息的字典
    """
    if verbose:
        print(f"\n开始测试推理速度 (迭代次数: {n_iterations})...")
    
    # 记录每次推理的时间
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []
    total_times = []
    
    for i in range(n_iterations):
        # 完整的推理流程
        start_total = time.perf_counter()
        
        # 1. 预处理
        start_pre = time.perf_counter()
        obs_processed = preprocessor(obs_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        preprocessing_time = time.perf_counter() - start_pre
        
        # 2. 推理
        start_inference = time.perf_counter()
        with torch.inference_mode():
            action = policy.predict_action_chunk(obs_processed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_inference
        
        # 3. 后处理
        start_post = time.perf_counter()
        action = postprocessor(action)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        postprocessing_time = time.perf_counter() - start_post
        
        total_time = time.perf_counter() - start_total
        
        # 记录时间
        preprocessing_times.append(preprocessing_time)
        inference_times.append(inference_time)
        postprocessing_times.append(postprocessing_time)
        total_times.append(total_time)
        
        # 每10次迭代打印一次进度
        if verbose and (i + 1) % 10 == 0:
            print(f"完成 {i + 1}/{n_iterations} 次迭代")
    
    # 计算统计信息
    stats = {
        "preprocessing": {
            "mean": np.mean(preprocessing_times) * 1000,  # 转换为毫秒
            "std": np.std(preprocessing_times) * 1000,
            "min": np.min(preprocessing_times) * 1000,
            "max": np.max(preprocessing_times) * 1000,
            "median": np.median(preprocessing_times) * 1000,
            "p95": np.percentile(preprocessing_times, 95) * 1000,
            "p99": np.percentile(preprocessing_times, 99) * 1000,
        },
        "inference": {
            "mean": np.mean(inference_times) * 1000,
            "std": np.std(inference_times) * 1000,
            "min": np.min(inference_times) * 1000,
            "max": np.max(inference_times) * 1000,
            "median": np.median(inference_times) * 1000,
            "p95": np.percentile(inference_times, 95) * 1000,
            "p99": np.percentile(inference_times, 99) * 1000,
        },
        "postprocessing": {
            "mean": np.mean(postprocessing_times) * 1000,
            "std": np.std(postprocessing_times) * 1000,
            "min": np.min(postprocessing_times) * 1000,
            "max": np.max(postprocessing_times) * 1000,
            "median": np.median(postprocessing_times) * 1000,
            "p95": np.percentile(postprocessing_times, 95) * 1000,
            "p99": np.percentile(postprocessing_times, 99) * 1000,
        },
        "total": {
            "mean": np.mean(total_times) * 1000,
            "std": np.std(total_times) * 1000,
            "min": np.min(total_times) * 1000,
            "max": np.max(total_times) * 1000,
            "median": np.median(total_times) * 1000,
            "p95": np.percentile(total_times, 95) * 1000,
            "p99": np.percentile(total_times, 99) * 1000,
        },
        "frequency_hz": 1.0 / np.mean(total_times),
    }
    
    return stats


def print_statistics(stats, policy_info: Dict[str, Any] = None):
    """
    打印统计信息
    
    Args:
        stats: 统计信息字典
        policy_info: policy相关信息
    """
    print("\n" + "="*70)
    print("推理速度测试结果")
    print("="*70)
    
    if policy_info:
        print(f"\nPolicy信息:")
        for key, value in policy_info.items():
            print(f"  {key}: {value}")
    
    print(f"\n预处理时间:")
    print(f"  平均: {stats['preprocessing']['mean']:.2f} ms")
    print(f"  标准差: {stats['preprocessing']['std']:.2f} ms")
    print(f"  中位数: {stats['preprocessing']['median']:.2f} ms")
    print(f"  最小: {stats['preprocessing']['min']:.2f} ms")
    print(f"  最大: {stats['preprocessing']['max']:.2f} ms")
    print(f"  P95: {stats['preprocessing']['p95']:.2f} ms")
    print(f"  P99: {stats['preprocessing']['p99']:.2f} ms")
    
    print(f"\n推理时间 (predict_action_chunk):")
    print(f"  平均: {stats['inference']['mean']:.2f} ms")
    print(f"  标准差: {stats['inference']['std']:.2f} ms")
    print(f"  中位数: {stats['inference']['median']:.2f} ms")
    print(f"  最小: {stats['inference']['min']:.2f} ms")
    print(f"  最大: {stats['inference']['max']:.2f} ms")
    print(f"  P95: {stats['inference']['p95']:.2f} ms")
    print(f"  P99: {stats['inference']['p99']:.2f} ms")
    
    print(f"\n后处理时间:")
    print(f"  平均: {stats['postprocessing']['mean']:.2f} ms")
    print(f"  标准差: {stats['postprocessing']['std']:.2f} ms")
    print(f"  中位数: {stats['postprocessing']['median']:.2f} ms")
    print(f"  最小: {stats['postprocessing']['min']:.2f} ms")
    print(f"  最大: {stats['postprocessing']['max']:.2f} ms")
    print(f"  P95: {stats['postprocessing']['p95']:.2f} ms")
    print(f"  P99: {stats['postprocessing']['p99']:.2f} ms")
    
    print(f"\n总时间 (端到端):")
    print(f"  平均: {stats['total']['mean']:.2f} ms")
    print(f"  标准差: {stats['total']['std']:.2f} ms")
    print(f"  中位数: {stats['total']['median']:.2f} ms")
    print(f"  最小: {stats['total']['min']:.2f} ms")
    print(f"  最大: {stats['total']['max']:.2f} ms")
    print(f"  P95: {stats['total']['p95']:.2f} ms")
    print(f"  P99: {stats['total']['p99']:.2f} ms")
    
    print(f"\n推理频率: {stats['frequency_hz']:.2f} Hz")
    print("="*70 + "\n")


def save_results(stats, policy_info: Dict[str, Any], output_path: str):
    """
    保存测试结果到JSON文件
    
    Args:
        stats: 统计信息字典
        policy_info: policy相关信息
        output_path: 输出文件路径
    """
    results = {
        "policy_info": policy_info,
        "statistics": stats,
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"测试结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="测试lerobot policy的推理速度")
    parser.add_argument(
        "--policy_path", 
        type=str, 
        default="/inspire/ssd/project/robot-decision/cengchendong-CZXS25230112/Projects/lerobot/outputs/pi05_0120/checkpoints/015000/pretrained_model",
        help="Policy checkpoint路径"
    )
    parser.add_argument(
        "--policy_type",
        type=str,
        choices=list(POLICY_CLASSES.keys()),
        default=None,
        help="Policy类型，如果不指定则自动检测"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="推理设备 (cuda:0, cuda:1, cpu等)"
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        default=True,
        help="禁用模型编译（默认禁用）"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--n_warmup", 
        type=int, 
        default=10,
        help="预热迭代次数"
    )
    parser.add_argument(
        "--n_iterations", 
        type=int, 
        default=100,
        help="测试迭代次数"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="保存测试结果的JSON文件路径"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，减少输出"
    )
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    if not args.quiet:
        print(f"使用设备: {device}")
    
    # 检测或使用指定的policy类型
    if args.policy_type is None:
        policy_type = detect_policy_type(args.policy_path)
        if not args.quiet:
            print(f"自动检测policy类型: {policy_type}")
    else:
        policy_type = args.policy_type
    
    PolicyClass = POLICY_CLASSES[policy_type]
    
    # 加载policy
    if not args.quiet:
        print(f"正在加载policy: {args.policy_path}")
    policy = PolicyClass.from_pretrained(args.policy_path).to(device).eval()
    
    # 通过config控制编译
    if args.no_compile and hasattr(policy.config, 'compile_model'):
        policy.config.compile_model = False
        if not args.quiet:
            print(f"已通过config禁用模型编译: compile_model={policy.config.compile_model}")
    elif not args.quiet and hasattr(policy.config, 'compile_model'):
        print(f"模型编译状态: compile_model={policy.config.compile_model}")
    if not args.quiet:
        print(f"Policy加载成功")
        print(f"Policy类型: {type(policy).__name__}")
        print(f"Action chunk size: {policy.config.n_action_steps}")
    
    # 收集policy信息
    policy_info = {
        "policy_path": args.policy_path,
        "policy_type": type(policy).__name__,
        "action_chunk_size": policy.config.n_action_steps,
        "device": str(device),
        "batch_size": args.batch_size,
    }
    
    # 创建预处理器和后处理器
    if not args.quiet:
        print("正在创建预处理器和后处理器...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        args.policy_path,
        preprocessor_overrides={
            "device_processor": {"device": str(device)}
        }
    )
    if not args.quiet:
        print("预处理器和后处理器创建成功")
    
    # 创建模拟观测数据
    if not args.quiet:
        print(f"正在创建模拟观测数据 (batch_size={args.batch_size})...")
    obs_dict = create_dummy_observation(policy, device, batch_size=args.batch_size)
    if not args.quiet:
        print(f"模拟观测数据创建成功")
        print(f"输入特征: {list(obs_dict.keys())}")
    
    # 预热
    warmup_inference(policy, preprocessor, obs_dict, n_warmup=args.n_warmup)
    
    # 测试推理速度
    stats = benchmark_inference(
        policy, 
        preprocessor, 
        postprocessor, 
        obs_dict, 
        n_iterations=args.n_iterations,
        verbose=not args.quiet
    )
    
    # 打印统计信息
    print_statistics(stats, policy_info=policy_info)
    
    # 保存结果到文件
    if args.output_file:
        save_results(stats, policy_info, args.output_file)
    
    if not args.quiet:
        print("测试完成!")


if __name__ == "__main__":
    main()
