"""
GCAM 简化版数据合成管线。

流程：
  参数采样 → GCAM仿真 → 质量验证 → 参数增强 → 统一参数转换 → HDF5存储

用法：
  python -m piern.simulators.gcam.pipeline \
      --config configs/gcam/variants/energy_transition.yaml
"""

import argparse
import logging
import os
import time
from typing import Tuple

import numpy as np
import yaml

from piern.simulators.gcam.generator import generate_batch
from piern.simulators.gcam.generator_with_params import generate_batch_from_params
from piern.simulators.gcam.unified_params import GCAMParamConverter
from piern.core.storage import save_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def filter_gcam_dataset(
    timeseries: np.ndarray,
    params: np.ndarray,
) -> Tuple:
    """
    GCAM 专用质量过滤。

    过滤条件：
    - 无 NaN/Inf
    - 份额在 [0, 1]
    - CO2排放为正
    - 温度在合理范围
    """
    if len(timeseries) == 0:
        return timeseries, params, {}

    valid_mask = np.ones(len(timeseries), dtype=bool)

    for i in range(len(timeseries)):
        ts = timeseries[i]
        if np.any(np.isnan(ts)) or np.any(np.isinf(ts)):
            valid_mask[i] = False
            continue
        # 份额检查
        if np.any(ts[0] < 0) or np.any(ts[0] > 1):
            valid_mask[i] = False
            continue
        if np.any(ts[1] < 0) or np.any(ts[1] > 1):
            valid_mask[i] = False
            continue
        # CO2排放
        if np.any(ts[2] < 0):
            valid_mask[i] = False
            continue
        # 温度
        if np.any(ts[4] < -1) or np.any(ts[4] > 8):
            valid_mask[i] = False
            continue

    n_valid = valid_mask.sum()
    stats = {'n_original': len(timeseries), 'n_valid': n_valid}
    return timeseries[valid_mask], params[valid_mask], stats


def perturb_params(
    params: np.ndarray,
    perturbation_ratio: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """对参数施加随机扰动。"""
    if rng is None:
        rng = np.random.default_rng()
    delta = rng.uniform(-perturbation_ratio, perturbation_ratio, size=params.shape)
    return (params * (1.0 + delta)).astype(np.float32)


def augment_with_parameter_sampling(
    original_timeseries: np.ndarray,
    original_params: np.ndarray,
    param_names: list,
    aug_cfg: dict,
    sim_cfg: dict,
    target_n: int,
    seed: int = 42,
    progress_callback=None,
) -> tuple:
    """通过参数空间采样增强 GCAM 数据集。"""
    rng = np.random.default_rng(seed)

    if not aug_cfg.get('enabled', True):
        return original_timeseries, original_params

    perturbation_ratio = aug_cfg.get('perturbation_ratio', 0.05)
    batch_size = aug_cfg.get('augmentation_batch_size', 200)

    pool_ts = list(original_timeseries)
    pool_params = list(original_params)

    if len(pool_ts) >= target_n:
        return original_timeseries, original_params

    logger.info(f"GCAM 参数增强：当前 {len(pool_ts)} → 配置数 {target_n}")

    round_idx = 0
    max_rounds = 50
    while len(pool_ts) < target_n and round_idx < max_rounds:
        round_idx += 1
        still_needed = target_n - len(pool_ts)
        this_batch = min(batch_size, still_needed * 2)

        pool_arr = np.array(pool_params)
        selected = pool_arr[rng.choice(len(pool_arr), size=this_batch, replace=True)]
        perturbed = perturb_params(selected, perturbation_ratio, rng)

        new_ts, new_params = generate_batch_from_params(perturbed, param_names, sim_cfg)
        if len(new_ts) == 0:
            continue

        new_ts_arr = np.stack(new_ts, axis=0)
        new_params_arr = np.array(new_params, dtype=np.float32)
        new_ts_arr, new_params_arr, _ = filter_gcam_dataset(new_ts_arr, new_params_arr)

        if len(new_ts_arr) == 0:
            continue

        pool_ts.extend(list(new_ts_arr))
        pool_params.extend(list(new_params_arr))
        logger.info(f"增强进度：{len(pool_ts)}/{target_n} 样本（第 {round_idx} 轮新增 {len(new_ts_arr)} 个）")
        if progress_callback:
            progress_callback(len(pool_ts), target_n)

    pool_ts = pool_ts[:target_n]
    pool_params = pool_params[:target_n]

    aug_ts = np.stack(pool_ts, axis=0)
    aug_params = np.array(pool_params, dtype=np.float32)

    idx = rng.permutation(len(aug_ts))
    return aug_ts[idx], aug_params[idx]


def run_pipeline(
    cfg_path: str,
    parallel: bool = False,
    max_workers: int = 4,
    n_samples: int = None,
    progress_callback=None,
) -> str:
    """
    执行完整 GCAM 数据合成管线。

    Returns:
        输出 HDF5 文件路径
    """
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    n_samples = n_samples if n_samples is not None else cfg['n_samples']
    seed = cfg.get('seed', 42)
    output_path = os.path.join(cfg['output_dir'], cfg['output_file'])
    output_stem = os.path.basename(cfg.get('output_file', 'unknown')).removesuffix('.h5')
    dir_name = os.path.basename(cfg.get('output_dir', ''))
    scenario_name = output_stem[len(dir_name) + 1:] if output_stem.startswith(dir_name + '_') else output_stem

    logger.info(f'===== GCAM 数据合成管线启动 =====')
    logger.info(f'场景: {scenario_name}')
    logger.info(f'配置样本数: {n_samples}')
    logger.info(f'输出路径: {output_path}')

    t0 = time.time()

    # Step 1: 生成种子数据
    aug_cfg = cfg.get('augmentation', {})
    seed_ratio = aug_cfg.get('seed_ratio', 0.2)
    n_seed = max(int(n_samples * seed_ratio), 50)
    logger.info(f'Step 1/4: 生成种子样本 ({n_seed} 个)...')
    timeseries, params, param_names = generate_batch(cfg, n_seed, seed=seed,
                                                     progress_callback=progress_callback,
                                                     progress_total=n_samples)
    logger.info(f'  → {timeseries.shape[0]} 个样本，形状 {timeseries.shape}')

    # Step 2: 质量过滤
    logger.info('Step 2/4: 质量过滤...')
    timeseries, params, stats = filter_gcam_dataset(timeseries, params)
    logger.info(f'  → 保留 {timeseries.shape[0]} 个样本（过滤 {stats["n_original"] - stats["n_valid"]} 个）')

    if timeseries.shape[0] == 0:
        raise RuntimeError('质量过滤后无有效样本')

    # Step 3: 参数增强
    if seed_ratio >= 1.0:
        logger.info('Step 3/4: seed_ratio=1.0，跳过增强')
        aug_ts = timeseries[:n_samples]
        aug_params = params[:n_samples]
    else:
        logger.info(f'Step 3/4: 参数增强（配置数 {n_samples} 个）...')
        aug_ts, aug_params = augment_with_parameter_sampling(
            timeseries, params, param_names, aug_cfg, cfg,
            target_n=n_samples, seed=seed + 1,
            progress_callback=progress_callback,
        )
    logger.info(f'  → 增强后 {aug_ts.shape[0]} 个样本')

    # Step 3.5: 转换统一参数
    logger.info('Step 3.5/4: 转换为18维统一参数...')
    converter = GCAMParamConverter()
    unified_list = []
    for i in range(len(aug_params)):
        params_dict = {name: float(aug_params[i, j]) for j, name in enumerate(param_names)}
        unified_list.append(converter.convert(scenario_name, params_dict))
    unified_params = np.array(unified_list, dtype=np.float32)
    logger.info(f'  → 参数维度: {len(param_names)}维 → 18维')

    # Step 4: 存储
    logger.info('Step 4/4: 写入 HDF5...')
    metadata = {
        'config_path': cfg_path,
        'scenario_name': scenario_name,
        'simulator': 'gcam_simplified',
        'n_original': timeseries.shape[0],
        'n_augmented': aug_ts.shape[0],
        'augmentation_method': 'parameter_sampling',
        'param_names': converter.param_names,
        'unified_params_version': 'v1.0',
        'original_param_names': param_names,
        'output_variables': ['coal_share', 'renewable_share', 'co2_emission', 'energy_price', 'temperature'],
        'time_axis': '2025-2100 (5yr steps)',
    }
    save_dataset(output_path, aug_ts, unified_params, converter.param_names, metadata)

    elapsed = time.time() - t0
    logger.info(f'===== 完成！耗时 {elapsed:.1f}s =====')
    logger.info(f'数据集形状: timeseries={aug_ts.shape}, params={unified_params.shape}')
    logger.info(f'输出文件: {output_path}')

    return output_path


def main():
    parser = argparse.ArgumentParser(description='GCAM 简化版数据合成管线')
    parser.add_argument('--config', type=str, required=True, help='YAML配置文件路径')
    parser.add_argument('--n-samples', type=int, default=None, help='配置样本数')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'配置文件不存在: {args.config}')

    run_pipeline(args.config, parallel=args.parallel, max_workers=args.max_workers, n_samples=args.n_samples)


if __name__ == '__main__':
    main()
