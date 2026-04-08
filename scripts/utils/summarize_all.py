"""汇总所有 Stage 1 数据的统计信息。"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List
import yaml


def analyze_h5_file(file_path: Path) -> Dict:
    """分析单个 HDF5 文件。"""
    with h5py.File(file_path, "r") as f:
        timeseries = f["timeseries"][:]
        params = f["params"][:]
        param_names = [n.decode("utf-8") for n in f["param_names"][:]]

        stats = {
            "file": file_path.name,
            "n_samples": timeseries.shape[0],
            "n_channels": timeseries.shape[1],   # 通用名（n_wells for MODFLOW, 43 for power, etc.）
            "n_wells": timeseries.shape[1],       # 保留向后兼容
            "n_timesteps": timeseries.shape[2],
            "n_params": params.shape[1],
            "file_size_mb": file_path.stat().st_size / 1024 / 1024,
            "param_ranges": {},
            "timeseries_stats": {
                "min": float(timeseries.min()),
                "max": float(timeseries.max()),
                "mean": float(timeseries.mean()),
                "nan_ratio": float(np.isnan(timeseries).mean()),
            },
        }

        # 参数范围
        for i, name in enumerate(param_names):
            values = params[:, i]
            stats["param_ranges"][name] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }

    return stats


def main():
    data_dir = Path("data/modflow")
    config_dir = Path("configs/data_synthesis/modflow_variants")

    # 查找所有 HDF5 文件
    h5_files = sorted(data_dir.glob("*_groundwater_timeseries.h5"))

    print("=" * 80)
    print("Stage 1 数据汇总")
    print("=" * 80)
    print()

    if not h5_files:
        print("未找到任何数据文件！")
        return

    print(f"发现 {len(h5_files)} 个数据文件")
    print()

    # 分析每个文件
    all_stats = []
    total_samples = 0
    total_size_mb = 0

    for h5_file in h5_files:
        try:
            stats = analyze_h5_file(h5_file)
            all_stats.append(stats)
            total_samples += stats["n_samples"]
            total_size_mb += stats["file_size_mb"]
        except Exception as e:
            print(f"✗ 分析失败: {h5_file.name} - {e}")

    # 打印汇总表格
    print("数据文件列表:")
    print("-" * 80)
    print(f"{'文件名':<45} {'样本数':>8} {'时间步':>8} {'观测井':>6} {'大小(MB)':>10}")
    print("-" * 80)

    for stats in all_stats:
        print(
            f"{stats['file']:<45} "
            f"{stats['n_samples']:>8} "
            f"{stats['n_timesteps']:>8} "
            f"{stats['n_wells']:>6} "
            f"{stats['file_size_mb']:>10.2f}"
        )

    print("-" * 80)
    print(f"{'总计':<45} {total_samples:>8} {'':<8} {'':<6} {total_size_mb:>10.2f}")
    print()

    # 参数覆盖范围
    print("参数覆盖范围（所有数据集）:")
    print("-" * 80)

    param_names = ["hk", "sy", "pumping", "strt", "rch"]
    global_ranges = {name: {"min": float("inf"), "max": float("-inf")} for name in param_names}

    for stats in all_stats:
        for name, ranges in stats["param_ranges"].items():
            if name in global_ranges:
                global_ranges[name]["min"] = min(global_ranges[name]["min"], ranges["min"])
                global_ranges[name]["max"] = max(global_ranges[name]["max"], ranges["max"])

    for name in param_names:
        r = global_ranges[name]
        span = r["max"] / r["min"] if r["min"] > 0 else 0
        print(f"  {name:10s}: [{r['min']:12.6f}, {r['max']:12.6f}]  跨度: {span:.1f}×")

    print()

    # 时序统计
    print("时序数据统计（所有数据集）:")
    print("-" * 80)

    all_head_min = min(s["timeseries_stats"]["min"] for s in all_stats)
    all_head_max = max(s["timeseries_stats"]["max"] for s in all_stats)
    avg_head_mean = np.mean([s["timeseries_stats"]["mean"] for s in all_stats])
    max_nan_ratio = max(s["timeseries_stats"]["nan_ratio"] for s in all_stats)

    print(f"  水头范围: [{all_head_min:.2f}, {all_head_max:.2f}] m")
    print(f"  平均水头: {avg_head_mean:.2f} m")
    print(f"  最大 NaN 比例: {max_nan_ratio * 100:.4f}%")
    print()

    # 配置覆盖情况
    print("配置覆盖情况:")
    print("-" * 80)

    config_files = list(config_dir.glob("*.yaml"))
    config_files = [c for c in config_files if c.stem != "README"]

    generated_configs = set(s["file"].replace("_groundwater_timeseries.h5", "") for s in all_stats)
    all_configs = set(c.stem for c in config_files)

    missing_configs = all_configs - generated_configs

    print(f"  总配置数: {len(all_configs)}")
    print(f"  已生成: {len(generated_configs)}")
    print(f"  未生成: {len(missing_configs)}")

    if missing_configs:
        print(f"\n  未生成的配置:")
        for config in sorted(missing_configs):
            print(f"    - {config}")

    print()
    print("=" * 80)
    print("汇总完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
