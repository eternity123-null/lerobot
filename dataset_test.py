from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id = "single_arm_v6"   # 或本地已有数据的 repo 名
root = "/inspire/hdd/project/robot-decision/public/datasets/single_arm_v6"  # 已经本地下载则填数据根目录；留 None 会使用默认 cache，并自动只下载 meta/

meta = LeRobotDatasetMetadata(repo_id, root=root, force_cache_sync=False)

# meta.episodes 是 HuggingFace Dataset，这里转成 pandas 方便 groupby
episodes_df = meta.episodes.to_pandas()
counts = (
    episodes_df[["episode_index", "tasks", "length"]]
    .explode("tasks")
    .groupby("tasks")
    .agg(
        episodes=("episode_index", "nunique"),
        total_frames=("length", "sum"),
    )
    .sort_values("episodes", ascending=False)
)

print(counts)