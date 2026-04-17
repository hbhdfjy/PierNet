"""
MODFLOW 数据合成管线 V2 - 支持参数空间采样增强。

改进点：
  1. 删除无物理意义的 Scaling/Offset 扰动
  2. 使用参数空间采样增强（在参数邻域采样 → 运行 MODFLOW）
  3. 保持参数-时序映射的物理一致性

流程：
  参数采样
    ↓
  MODFLOW 正演（generate_batch）
    ↓
  质量过滤（filter_dataset）
    ↓
  参数空间采样增强（循环，直到总样本数达到 n_samples）
    ├─ 每轮从当前样本池中随机选取一批
    ├─ ±5% 扰动参数 → 运行 MODFLOW
    ├─ 合并到样本池
    └─ 重复直到达到目标数量
    ↓
  HDF5 存储（save_dataset）

用法：
  python -m piern.simulators.modflow.pipeline \
      --config configs/modflow/default.yaml
"""

import argparse
import logging
import os
import time

import numpy as np
import yaml

from piern.simulators.modflow.generator import generate_batch, resolve_modflow_executable
from piern.simulators.modflow.generator_with_params import generate_batch_from_params
from piern.simulators.modflow.unified_params import UnifiedParamConverter
from piern.core.validation import filter_dataset
from piern.core.storage import save_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _compute_seed_batch_size(n_samples: int, seed_ratio: float, minimum_seed: int) -> int:
    requested = max(int(np.ceil(n_samples * max(seed_ratio, 0.0))), 1)
    if n_samples <= minimum_seed:
        return max(1, n_samples)
    return max(requested, minimum_seed)


def perturb_params(
    params: np.ndarray,
    param_names: list[str],
    perturbation_ratio: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    在参数空间中对参数进行小幅扰动。

    Args:
        params: 原始参数，形状 [N, n_params]
        param_names: 参数名称列表
        perturbation_ratio: 扰动比例（相对于参数值）
        rng: 随机数生成器

    Returns:
        扰动后的参数，形状 [N, n_params]
    """
    if rng is None:
        rng = np.random.default_rng()

    N, n_params = params.shape

    # 为每个参数生成扰动因子：1 + δ，δ ~ Uniform[-r, r]
    delta = rng.uniform(-perturbation_ratio, perturbation_ratio, size=(N, n_params))
    perturbed_params = params * (1.0 + delta)

    return perturbed_params.astype(np.float32)


def augment_with_parameter_sampling(
    original_timeseries: np.ndarray,
    original_params: np.ndarray,
    param_names: list[str],
    aug_cfg: dict,
    modflow_cfg: dict,
    target_n: int,
    seed: int = 42,
    max_workers: int = 1,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    通过参数空间采样增强数据集，循环直到总样本数达到 target_n。

    每轮从当前样本池中随机选取一批，施加 ±perturbation_ratio 扰动后跑
    MODFLOW，将成功的新样本追加到样本池，重复直到达到目标。

    Args:
        original_timeseries: 原始时序，[N, n_wells, n_timesteps]
        original_params: 原始参数，[N, n_params]
        param_names: 参数名称列表
        aug_cfg: 增强配置
        modflow_cfg: MODFLOW 配置
        target_n: 目标总样本数
        seed: 随机种子

    Returns:
        aug_timeseries: [target_n, n_wells, n_timesteps]
        aug_params: [target_n, n_params]
    """
    rng = np.random.default_rng(seed)

    # 检查增强是否启用
    if not aug_cfg.get("enabled", True):
        logger.info("数据增强已禁用，返回原始数据")
        return original_timeseries, original_params

    perturbation_ratio = aug_cfg.get("perturbation_ratio", 0.05)
    # 每轮批量大小：每次尝试生成这么多扰动样本
    batch_size = aug_cfg.get("augmentation_batch_size", 1000)

    # 当前样本池
    pool_ts = list(original_timeseries)
    pool_params = list(original_params)

    N_current = len(pool_ts)
    N_needed = target_n - N_current

    if N_needed <= 0:
        logger.info(f"当前样本数 {N_current} 已达到配置数 {target_n}，跳过增强")
        return original_timeseries, original_params

    logger.info(f"参数空间采样增强：当前 {N_current} 个样本，配置数 {target_n}，需补充 {N_needed} 个")
    logger.info(f"  - 扰动比例: ±{perturbation_ratio*100:.1f}%，每轮批量: {batch_size}")

    round_idx = 0
    while len(pool_ts) < target_n:
        round_idx += 1
        still_needed = target_n - len(pool_ts)
        this_batch = min(batch_size, still_needed * 2)  # 多请求一些，抵消失败率

        # 从当前样本池中有放回地随机选取
        pool_arr = np.array(pool_params)
        selected_indices = rng.choice(len(pool_arr), size=this_batch, replace=True)
        selected_params = pool_arr[selected_indices]

        # 施加扰动
        perturbed = perturb_params(
            selected_params,
            param_names,
            perturbation_ratio=perturbation_ratio,
            rng=rng,
        )
        # clip 到配置范围，防止扰动后超出设计边界
        p_cfg = modflow_cfg.get('params', {})
        for j, name in enumerate(param_names):
            lo = p_cfg.get(f'{name}_min')
            hi = p_cfg.get(f'{name}_max')
            if lo is not None and hi is not None:
                perturbed[:, j] = np.clip(perturbed[:, j], lo, hi)

        # 跑 MODFLOW（并行时逐任务报进度）
        if max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            from piern.simulators.modflow.generator import _generate_single_sample_worker
            # 直接构建任务列表，内联种子生成（无需中间 tasks 变量）
            base_seed = int(rng.integers(0, 2**31))
            tasks_with_params = []
            for i, param_row in enumerate(perturbed):
                cfg_copy = dict(modflow_cfg)
                cfg_copy['_override_params'] = {n: float(param_row[j]) for j, n in enumerate(param_names)}
                tasks_with_params.append((cfg_copy, base_seed + i))

            from concurrent.futures import as_completed
            new_ts_raw, new_params_raw = [], []
            done_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_generate_single_sample_worker, t) for t in tasks_with_params]
                for future in as_completed(futures):
                    if len(pool_ts) + len(new_ts_raw) >= target_n:
                        for f in futures:
                            f.cancel()
                        break
                    done_count += 1
                    try:
                        ts, p = future.result()
                    except Exception:
                        continue
                    if ts is not None and p is not None:
                        new_ts_raw.append(ts)
                        new_params_raw.append(p)
                    if done_count % max(max_workers, 1) == 0:
                        current = len(pool_ts) + len(new_ts_raw)
                        logger.info(f"增强进度：{current}/{target_n} 样本")
                        if progress_callback:
                            progress_callback(current, target_n)
            new_ts = new_ts_raw
            new_params = new_params_raw
        else:
            new_ts, new_params = generate_batch_from_params(
                perturbed,
                param_names,
                modflow_cfg,
            )

        if len(new_ts) == 0:
            logger.warning(f"  第 {round_idx} 轮：所有扰动样本生成失败，跳过")
            continue

        # 质量过滤（过滤干涸样本等）
        new_ts_arr = np.stack(new_ts, axis=0)
        new_params_arr = np.array(new_params, dtype=np.float32)
        new_ts_arr, new_params_arr, _ = filter_dataset(new_ts_arr, new_params_arr, modflow_cfg.get("validation", {}))
        if len(new_ts_arr) == 0:
            logger.warning(f"  第 {round_idx} 轮：质量过滤后无有效样本，跳过")
            continue

        # 追加到样本池
        pool_ts.extend(list(new_ts_arr))
        pool_params.extend(list(new_params_arr))

        logger.info(
            f"  第 {round_idx} 轮：新增 {len(new_ts)}/{this_batch} 个样本"
            f"（成功率 {len(new_ts)/this_batch*100:.1f}%），"
            f"当前总计 {len(pool_ts)}/{target_n}"
        )

    # 截取到目标数量并打乱
    pool_ts = pool_ts[:target_n]
    pool_params = pool_params[:target_n]

    aug_timeseries = np.stack(pool_ts, axis=0)
    aug_params = np.array(pool_params, dtype=np.float32)

    shuffle_idx = rng.permutation(target_n)
    aug_timeseries = aug_timeseries[shuffle_idx]
    aug_params = aug_params[shuffle_idx]

    logger.info(
        f"增强完成：原始 {len(original_timeseries)} → 总计 {target_n} 个样本"
        f"（共 {round_idx} 轮）"
    )

    return aug_timeseries, aug_params


def run_pipeline(cfg_path: str, parallel: bool = False, max_workers: int = 10, n_samples: int | None = None, progress_callback=None) -> str:
    """
    执行完整合成管线（V2 版本 + 统一参数表示）。

    Args:
        cfg_path: YAML 配置文件路径
        parallel: 是否使用并行生成（单场景内并行）
        max_workers: 并行进程数
        n_samples: 配置样本数（覆盖配置文件中的 n_samples）

    Returns:
        输出 HDF5 文件路径
    """
    # 加载配置
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 命令行参数优先，其次读配置文件
    n_samples = n_samples if n_samples is not None else cfg["n_samples"]
    seed = cfg.get("seed", 42)
    output_path = os.path.join(cfg["output_dir"], cfg["output_file"])

    # 提取场景名称：去掉 .h5 后缀，再去掉 "大场景_" 前缀（如 modflow_unified_aquifer → unified_aquifer）
    output_stem = os.path.basename(cfg.get("output_file", "unknown")).removesuffix(".h5")
    dir_name = os.path.basename(cfg.get("output_dir", ""))
    scenario_name = output_stem[len(dir_name) + 1:] if output_stem.startswith(dir_name + "_") else output_stem

    logger.info(f"===== MODFLOW 数据合成管线 V2 启动 =====")
    logger.info(f"场景名称: {scenario_name}")
    logger.info(f"配置样本数: {n_samples}")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"增强方法: 参数空间采样")
    logger.info(f"参数表示: 18维统一参数")
    modflow_exe = resolve_modflow_executable(cfg)
    logger.info(f"MODFLOW ?????: {modflow_exe}")
    if parallel:
        logger.info(f"🚀 并行模式: {max_workers} 个进程")

    t0 = time.time()

    # Step 1: 生成原始种子数据（只需生成目标数量的一部分作为起点）
    aug_cfg = cfg.get("augmentation", {})
    seed_ratio = aug_cfg.get("seed_ratio", 0.1)  # 默认先生成 10% 作为种子
    n_seed = _compute_seed_batch_size(n_samples, seed_ratio, minimum_seed=100)
    logger.info(f"Step 1/4: 运行 MODFLOW 正演（种子批次 {n_seed} 个）...")
    timeseries, params, param_names = generate_batch(
        cfg, n_seed, seed=seed, parallel=parallel, max_workers=max_workers,
        progress_callback=progress_callback, progress_total=n_samples,
    )
    logger.info(f"  → 生成 {timeseries.shape[0]} 个样本，形状 {timeseries.shape}")

    # Step 2: 质量过滤
    logger.info("Step 2/4: 质量过滤...")
    timeseries, params, _ = filter_dataset(timeseries, params, cfg["validation"])
    logger.info(f"  → 过滤后保留 {timeseries.shape[0]} 个样本")

    if timeseries.shape[0] == 0:
        raise RuntimeError("质量过滤后无有效样本，请检查 MODFLOW 配置或验证阈值")

    # Step 3: 补充采样 / 参数空间增强
    if timeseries.shape[0] >= n_samples:
        # 已有足够样本，直接截取
        logger.info("Step 3/4: 直接采样已满足配置数量，跳过增强")
        aug_ts = timeseries[:n_samples]
        aug_params = params[:n_samples]
    elif seed_ratio >= 1.0:
        # seed_ratio=1.0：不做参数扰动，继续用真实物理采样补充不足部分
        still_needed = n_samples - timeseries.shape[0]
        logger.info(f"Step 3/4: 直接采样不足，继续补充 {still_needed} 个样本（真实采样）...")
        extra_ts_list = [timeseries]
        extra_params_list = [params]
        batch_seed = seed + 1000
        round_idx = 0
        while sum(x.shape[0] for x in extra_ts_list) < n_samples:
            round_idx += 1
            needed = n_samples - sum(x.shape[0] for x in extra_ts_list)
            batch_n = needed if n_samples <= 100 else max(needed * 3, 100)
            logger.info(f"  补充第 {round_idx} 轮：请求 {batch_n} 个，还差 {needed} 个")
            extra_ts, extra_p, _ = generate_batch(
                cfg, batch_n, seed=batch_seed, parallel=parallel, max_workers=max_workers,
                progress_callback=progress_callback, progress_total=n_samples,
            )
            batch_seed += batch_n
            extra_ts, extra_p, _ = filter_dataset(extra_ts, extra_p, cfg["validation"])
            if extra_ts.shape[0] > 0:
                extra_ts_list.append(extra_ts)
                extra_params_list.append(extra_p)
                current = sum(x.shape[0] for x in extra_ts_list)
                logger.info(f"  增强进度：{current}/{n_samples} 样本")
                if progress_callback:
                    progress_callback(current, n_samples)
            if round_idx > 20:
                logger.warning("补充轮数超过 20 轮，提前终止")
                break
        all_ts = np.concatenate(extra_ts_list, axis=0)[:n_samples]
        all_params = np.concatenate(extra_params_list, axis=0)[:n_samples]
        # 打乱
        rng = np.random.default_rng(seed + 2)
        idx = rng.permutation(len(all_ts))
        aug_ts = all_ts[idx]
        aug_params = all_params[idx]
    else:
        logger.info(f"Step 3/4: 参数空间采样增强（配置数 {n_samples} 个）...")
        aug_ts, aug_params = augment_with_parameter_sampling(
            timeseries, params, param_names, aug_cfg, cfg,
            target_n=n_samples, seed=seed + 1, max_workers=max_workers,
            progress_callback=progress_callback,
        )
    logger.info(f"  → 增强后共 {aug_ts.shape[0]} 个样本")

    # Step 3.5: 转换为统一参数表示
    logger.info("Step 3.5/5: 转换为统一参数表示...")
    converter = UnifiedParamConverter()

    # 将原始参数（5维或更多）转换为统一参数（18维）
    unified_params_list = []
    for i in range(len(aug_params)):
        # 构建原始参数字典
        original_params_dict = {name: float(aug_params[i, j]) for j, name in enumerate(param_names)}

        # 转换为统一参数
        unified_params = converter.convert(scenario_name, original_params_dict)
        unified_params_list.append(unified_params)

    unified_params_array = np.array(unified_params_list, dtype=np.float32)
    unified_param_names = converter.param_names

    logger.info(f"  → 参数维度: {len(param_names)}维 → {len(unified_param_names)}维")
    logger.info(f"  → 场景类型: {int(unified_params_array[0, 15])}")
    logger.info(f"  → 输出类型: {int(unified_params_array[0, 16])}")

    # Step 4: 存储
    logger.info("Step 4/5: 写入 HDF5...")
    metadata = {
        "config_path": cfg_path,
        "scenario_name": scenario_name,
        "n_original": timeseries.shape[0],
        "n_augmented": aug_ts.shape[0],
        "augmentation_method": "parameter_sampling",
        "augmentation_config": cfg["augmentation"],
        "param_names": unified_param_names,  # 使用统一参数名
        "unified_params_version": "v1.0",
        "original_param_names": param_names,  # 保留原始参数名用于参考
    }
    save_dataset(output_path, aug_ts, unified_params_array, unified_param_names, metadata)

    elapsed = time.time() - t0
    logger.info(f"===== 完成！耗时 {elapsed:.1f}s =====")
    logger.info(f"数据集形状: timeseries={aug_ts.shape}, params={unified_params_array.shape}")
    logger.info(f"增强比例: {(aug_ts.shape[0] - timeseries.shape[0]) / timeseries.shape[0] * 100:.1f}%")
    logger.info(f"参数表示: 18维统一参数")
    logger.info(f"输出文件: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="MODFLOW 地下水位时序数据合成管线 V2")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_synthesis/modflow.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="配置样本数（覆盖配置文件中的 n_samples）",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用并行生成（单场景内并行）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="并行进程数（默认10）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")

    run_pipeline(args.config, parallel=args.parallel, max_workers=args.max_workers, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
