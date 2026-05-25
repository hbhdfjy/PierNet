"""
SimPEG 地球物理数据合成管线。

流程：
  参数采样 → 正演 → 质量过滤 → 参数空间采样增强 → 统一参数转换 → HDF5存储

用法：
  python -m PierNet.simulators.simpeg.pipeline \
      --config configs/simpeg/variants/dc_resistivity.yaml --n-samples 100
"""

import argparse
import logging
import os
import time

import numpy as np
import yaml

from PierNet.simulators.naming import scenario_name_from_output
from PierNet.simulators.simpeg.generator import generate_batch
from PierNet.simulators.simpeg.generator_with_params import generate_batch_from_params
from PierNet.simulators.simpeg.unified_params import SimPEGParamConverter
from PierNet.core.storage import save_dataset

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


def _top_up_direct_samples(
    cfg: dict,
    timeseries: np.ndarray,
    params: np.ndarray,
    n_samples: int,
    seed: int,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    batches_ts = [timeseries]
    batches_params = [params]
    batch_seed = seed + 1000
    round_idx = 0

    while sum(x.shape[0] for x in batches_ts) < n_samples:
        round_idx += 1
        current = sum(x.shape[0] for x in batches_ts)
        needed = n_samples - current
        batch_n = needed if n_samples <= 50 else max(needed * 2, 50)
        logger.info(f'  补充轮次 {round_idx}：再生成 {batch_n} 个候选，目标补足 {needed} 个')
        extra_ts, extra_params, _ = generate_batch(
            cfg, batch_n, seed=batch_seed,
            progress_callback=progress_callback,
            progress_total=n_samples,
        )
        batch_seed += batch_n
        extra_ts, extra_params, _ = filter_simpeg_dataset(extra_ts, extra_params, cfg)
        if extra_ts.shape[0] > 0:
            batches_ts.append(extra_ts)
            batches_params.append(extra_params)
            current = sum(x.shape[0] for x in batches_ts)
            logger.info(f'  当前累计有效样本 {current}/{n_samples} 条')
            if progress_callback:
                progress_callback(current, n_samples)
        else:
            logger.warning(f'  第 {round_idx} 轮没有产生有效样本')
        if round_idx >= 20:
            break

    all_ts = np.concatenate(batches_ts, axis=0)
    all_params = np.concatenate(batches_params, axis=0)
    if all_ts.shape[0] < n_samples:
        raise RuntimeError(f'SimPEG 有效样本不足，仅得到 {all_ts.shape[0]}/{n_samples} 条')

    all_ts = all_ts[:n_samples]
    all_params = all_params[:n_samples]
    idx = np.random.default_rng(seed + 2).permutation(n_samples)
    return all_ts[idx], all_params[idx]


def filter_simpeg_dataset(
    timeseries: np.ndarray,
    params: np.ndarray,
    cfg: dict,
) -> tuple:
    """
    SimPEG 专用质量过滤。

    过滤条件：
    - 无 NaN/Inf
    - 数值在合理范围内（正值，无极端值）
    - 最小方差（避免全零输出）
    """
    if len(timeseries) == 0:
        return timeseries, params, {}

    valid_mask = np.ones(len(timeseries), dtype=bool)
    val_cfg = cfg.get("validation", {})
    min_variance = val_cfg.get("min_variance", 1e-12)

    for i in range(len(timeseries)):
        ts = timeseries[i]  # [1, n_points]

        # NaN/Inf 检查
        if not np.isfinite(ts).all():
            valid_mask[i] = False
            continue

        # 正值检查（视电阻率、EMF等必须为正）
        if np.any(ts < 0):
            valid_mask[i] = False
            continue

        # 最小方差（避免常数输出）
        if ts.var() < min_variance:
            valid_mask[i] = False
            continue

    n_valid = valid_mask.sum()
    stats = {
        "n_original": len(timeseries),
        "n_valid": n_valid,
        "filter_ratio": 1.0 - n_valid / max(len(timeseries), 1),
    }
    return timeseries[valid_mask], params[valid_mask], stats


def perturb_params(
    params: np.ndarray,
    perturbation_ratio: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """对参数施加 ±perturbation_ratio 的随机扰动。"""
    if rng is None:
        rng = np.random.default_rng()
    delta = rng.uniform(-perturbation_ratio, perturbation_ratio, size=params.shape)
    return (params * (1.0 + delta)).astype(np.float32)


def augment_with_parameter_sampling(
    original_timeseries: np.ndarray,
    original_params: np.ndarray,
    param_names: list,
    aug_cfg: dict,
    simpeg_cfg: dict,
    target_n: int,
    seed: int = 42,
    progress_callback=None,
) -> tuple:
    """
    通过参数空间采样增强数据集，循环直到总样本数达到 target_n。
    """
    rng = np.random.default_rng(seed)

    if not aug_cfg.get("enabled", True):
        logger.info("数据增强已禁用，返回原始数据")
        return original_timeseries, original_params

    perturbation_ratio = aug_cfg.get("perturbation_ratio", 0.05)
    batch_size = aug_cfg.get("augmentation_batch_size", 500)

    pool_ts = list(original_timeseries)
    pool_params = list(original_params)

    N_current = len(pool_ts)
    if N_current >= target_n:
        logger.info(f"当前样本数 {N_current} 已达到配置数 {target_n}，跳过增强")
        return original_timeseries, original_params

    logger.info(f"参数空间采样增强：当前 {N_current} → 配置数 {target_n}")

    round_idx = 0
    max_rounds = 50
    while len(pool_ts) < target_n and round_idx < max_rounds:
        round_idx += 1
        still_needed = target_n - len(pool_ts)
        this_batch = min(batch_size, still_needed * 2)

        pool_arr = np.array(pool_params)
        selected = rng.choice(len(pool_arr), size=this_batch, replace=True)
        selected_params = pool_arr[selected]

        perturbed = perturb_params(selected_params, perturbation_ratio, rng)
        # clip 到配置范围，防止扰动后超出设计边界
        p_cfg = simpeg_cfg.get('params', {})
        for j, name in enumerate(param_names):
            lo = p_cfg.get(f'{name}_min')
            hi = p_cfg.get(f'{name}_max')
            if lo is not None and hi is not None:
                perturbed[:, j] = np.clip(perturbed[:, j], lo, hi)
        new_ts, new_params = generate_batch_from_params(perturbed, param_names, simpeg_cfg)

        if len(new_ts) == 0:
            logger.warning(f"  第 {round_idx} 轮：所有扰动样本生成失败，跳过")
            continue

        new_ts_arr = np.stack(new_ts, axis=0)
        new_params_arr = np.array(new_params, dtype=np.float32)
        new_ts_arr, new_params_arr, _ = filter_simpeg_dataset(new_ts_arr, new_params_arr, simpeg_cfg)

        if len(new_ts_arr) == 0:
            continue

        pool_ts.extend(list(new_ts_arr))
        pool_params.extend(list(new_params_arr))
        logger.info(
            f"  第 {round_idx} 轮：新增 {len(new_ts_arr)}/{this_batch} 个样本，"
            f"当前总计 {len(pool_ts)}/{target_n}"
        )
        if progress_callback:
            progress_callback(len(pool_ts), target_n)

    pool_ts = pool_ts[:target_n]
    pool_params = pool_params[:target_n]

    aug_timeseries = np.stack(pool_ts, axis=0)
    aug_params = np.array(pool_params, dtype=np.float32)

    shuffle_idx = rng.permutation(target_n)
    return aug_timeseries[shuffle_idx], aug_params[shuffle_idx]


def run_pipeline(cfg_path: str, n_samples: int = None, progress_callback=None) -> str:
    """
    执行完整 SimPEG 数据合成管线。

    Args:
        cfg_path:  YAML 配置文件路径
        n_samples: 配置样本数（覆盖配置文件中的值）

    Returns:
        输出 HDF5 文件路径
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    n_samples = n_samples if n_samples is not None else cfg["n_samples"]
    seed = cfg.get("seed", 42)
    output_path = os.path.join(cfg["output_dir"], cfg["output_file"])
    scenario_name = scenario_name_from_output(cfg.get("output_dir"), cfg.get("output_file"))

    logger.info("===== SimPEG 数据合成管线启动 =====")
    logger.info(f"场景名称: {scenario_name}")
    logger.info(f"配置样本数: {n_samples}")
    logger.info(f"输出路径: {output_path}")

    t0 = time.time()

    # Step 1: 生成种子数据
    aug_cfg = cfg.get("augmentation", {})
    seed_ratio = aug_cfg.get("seed_ratio", 1.0)
    n_seed = _compute_seed_batch_size(n_samples, seed_ratio, minimum_seed=50)
    logger.info(f"Step 1/4: SimPEG 正演（种子批次 {n_seed} 个）...")
    timeseries, params, param_names = generate_batch(cfg, n_seed, seed=seed,
                                                     progress_callback=progress_callback,
                                                     progress_total=n_samples)
    logger.info(f"  → 生成 {timeseries.shape[0]} 个样本，形状 {timeseries.shape}")

    # Step 2: 质量过滤
    logger.info("Step 2/4: 质量过滤...")
    timeseries, params, filter_stats = filter_simpeg_dataset(timeseries, params, cfg)
    logger.info(f"  → 过滤后保留 {timeseries.shape[0]} 个样本（过滤率 {filter_stats.get('filter_ratio', 0)*100:.1f}%）")

    if timeseries.shape[0] == 0:
        raise RuntimeError("质量过滤后无有效样本，请检查 SimPEG 配置或验证阈值")

    # Step 3: 参数空间采样增强
    if timeseries.shape[0] >= n_samples:
        logger.info("Step 3/4: 直接采样已满足配置数量，跳过增强")
        aug_ts = timeseries[:n_samples]
        aug_params = params[:n_samples]
    elif seed_ratio >= 1.0 or not aug_cfg.get("enabled", True):
        logger.info("Step 3/4: 直接采样不足，继续补充真实样本")
        aug_ts, aug_params = _top_up_direct_samples(
            cfg, timeseries, params, n_samples, seed, progress_callback=progress_callback
        )
    else:
        logger.info(f"Step 3/4: 参数空间采样增强（配置数 {n_samples} 个）...")
        aug_ts, aug_params = augment_with_parameter_sampling(
            timeseries, params, param_names, aug_cfg, cfg,
            target_n=n_samples, seed=seed + 1,
            progress_callback=progress_callback,
        )
    logger.info(f"  → 增强后共 {aug_ts.shape[0]} 个样本")
    if aug_ts.shape[0] < n_samples:
        raise RuntimeError(f"SimPEG 有效样本不足，仅得到 {aug_ts.shape[0]}/{n_samples} 条")

    # Step 3.5: 转换为统一参数
    logger.info("Step 3.5/5: 转换为统一参数表示...")
    converter = SimPEGParamConverter()
    unified_params_list = []
    for i in range(len(aug_params)):
        original_params_dict = {name: float(aug_params[i, j]) for j, name in enumerate(param_names)}
        unified = converter.convert(scenario_name, original_params_dict)
        unified_params_list.append(unified)

    unified_params_array = np.array(unified_params_list, dtype=np.float32)
    unified_param_names = converter.param_names

    logger.info(f"  → 参数维度: {len(param_names)}维 → {len(unified_param_names)}维")

    # Step 4: 存储
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Step 4/5: 写入 HDF5...")
    metadata = {
        "config_path": cfg_path,
        "scenario_name": scenario_name,
        "n_original": timeseries.shape[0],
        "n_augmented": aug_ts.shape[0],
        "augmentation_method": "parameter_sampling",
        "param_names": unified_param_names,
        "unified_params_version": "v1.0",
        "original_param_names": param_names,
        "min_head_value": 0.0,   # 满足 filter_dataset 的 key 要求
        "max_head_value": 1e6,
    }
    save_dataset(output_path, aug_ts, unified_params_array, unified_param_names, metadata)

    elapsed = time.time() - t0
    logger.info(f"===== 完成！耗时 {elapsed:.1f}s =====")
    logger.info(f"数据集形状: timeseries={aug_ts.shape}, params={unified_params_array.shape}")
    logger.info(f"输出文件: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="SimPEG 地球物理数据合成管线")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--n-samples", type=int, default=None, help="配置样本数")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")

    run_pipeline(args.config, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
