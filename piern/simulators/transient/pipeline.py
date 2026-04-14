"""
暂态稳定数据合成管线（ANDES）。

流程：
  参数采样 → 暂态稳定仿真（DAE）→ 质量过滤 → 参数增强 → 统一参数转换 → HDF5 存储

用法：
  python -m piern.simulators.transient.pipeline \
      --config configs/transient/variants/ieee14_fault.yaml --n-samples 500
"""

import argparse
import logging
import os
import time

import numpy as np
import yaml

from piern.simulators.transient.generator import generate_batch, _generate_single_worker
from piern.simulators.power_system.generator_with_params import generate_batch_from_params
from piern.simulators.power_system.unified_params import PowerSystemParamConverter
from piern.core.validation import filter_dataset
from piern.core.storage import save_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def _make_validation_cfg(sim_cfg: dict) -> dict:
    """构建暂态验证配置：转子角（归一化后）应在 [-3, 3] 范围内。"""
    vcfg = dict(sim_cfg.get('validation', {}))
    vcfg.setdefault('max_nan_ratio', 0.05)
    vcfg.setdefault('min_variance', 1e-8)
    vcfg.setdefault('min_head_value', -3.0)
    vcfg.setdefault('max_head_value', 3.0)
    return vcfg


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
    sim_cfg: dict,
    target_n: int,
    seed: int = 42,
    max_workers: int = 1,
    progress_callback=None,
) -> tuple:
    """通过参数空间采样增强数据集。max_workers>1 时使用多线程并行。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rng = np.random.default_rng(seed)

    if not aug_cfg.get('enabled', True):
        return original_timeseries, original_params

    perturbation_ratio = aug_cfg.get('perturbation_ratio', 0.05)
    batch_size = aug_cfg.get('augmentation_batch_size', 500)

    pool_ts = list(original_timeseries)
    pool_params = list(original_params)

    N_current = len(pool_ts)
    if N_current >= target_n:
        return original_timeseries, original_params

    logger.info(f"参数增强：当前 {N_current} → 目标 {target_n}（并行={max_workers}核）")

    validation_cfg = _make_validation_cfg(sim_cfg)

    round_idx = 0
    seed_offset = seed * 1000
    max_rounds = 50
    while len(pool_ts) < target_n and round_idx < max_rounds:
        round_idx += 1
        still_needed = target_n - len(pool_ts)
        this_batch = min(batch_size, still_needed * 2)

        pool_arr = np.array(pool_params)
        selected = pool_arr[rng.choice(len(pool_arr), size=this_batch, replace=True)]
        perturbed = perturb_params(selected, perturbation_ratio, rng)

        if max_workers > 1:
            tasks = []
            for i, param_row in enumerate(perturbed):
                override_cfg = dict(sim_cfg)
                override_cfg['_override_params'] = {
                    name: float(param_row[j]) for j, name in enumerate(param_names)
                }
                tasks.append((override_cfg, seed_offset + round_idx * 10000 + i))
            seed_offset += this_batch

            new_ts_raw = []
            new_params_raw = []
            done_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_generate_single_worker, t) for t in tasks]
                for future in as_completed(futures):
                    if len(pool_ts) + len(new_ts_raw) >= target_n:
                        for f in futures:
                            f.cancel()
                        break
                    done_count += 1
                    try:
                        ts, params = future.result()
                    except Exception:
                        continue
                    if ts is not None and params is not None:
                        row = [params.get(k, float('nan')) for k in param_names]
                        if not any(np.isnan(v) for v in row):
                            new_ts_raw.append(ts)
                            new_params_raw.append(row)
                    if done_count % max(max_workers, 1) == 0:
                        current = len(pool_ts) + len(new_ts_raw)
                        logger.info(f"增强进度：{current}/{target_n} 样本")
                        if progress_callback:
                            progress_callback(current, target_n)
            new_ts = new_ts_raw
            new_params = [np.array(p, dtype=np.float32) for p in new_params_raw]
        else:
            new_ts, new_params = generate_batch_from_params(perturbed, param_names, sim_cfg)

        if len(new_ts) == 0:
            logger.warning(f"第 {round_idx} 轮：增强失败，跳过")
            continue

        new_ts_arr = np.stack(new_ts, axis=0)
        new_params_arr = np.array(new_params, dtype=np.float32)
        new_ts_arr, new_params_arr, _ = filter_dataset(new_ts_arr, new_params_arr, validation_cfg)
        if len(new_ts_arr) == 0:
            continue

        pool_ts.extend(list(new_ts_arr))
        pool_params.extend(list(new_params_arr))
        logger.info(f"第 {round_idx} 轮：新增 {len(new_ts_arr)} 个，当前 {len(pool_ts)}/{target_n}")

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
    执行完整暂态稳定数据合成管线。

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

    logger.info('===== 暂态稳定数据合成管线启动（ANDES）=====')
    logger.info(f'场景: {scenario_name}')
    logger.info(f'目标样本数: {n_samples}')
    logger.info(f'输出路径: {output_path}')

    t0 = time.time()

    aug_cfg = cfg.get('augmentation', {})
    seed_ratio = aug_cfg.get('seed_ratio', 0.1)
    n_seed = max(int(n_samples * seed_ratio), 50)
    logger.info(f'Step 1/5: 生成种子样本 ({n_seed} 个)...')
    timeseries, params, param_names = generate_batch(
        cfg, n_seed, seed=seed, parallel=parallel, max_workers=max_workers,
        progress_callback=progress_callback, progress_total=n_samples,
    )
    logger.info(f'  → {timeseries.shape[0]} 个样本，形状 {timeseries.shape}')

    logger.info('Step 2/5: 质量过滤...')
    validation_cfg = _make_validation_cfg(cfg)
    timeseries, params, _ = filter_dataset(timeseries, params, validation_cfg)
    logger.info(f'  → 保留 {timeseries.shape[0]} 个样本')

    if timeseries.shape[0] == 0:
        raise RuntimeError('质量过滤后无有效样本')

    if seed_ratio >= 1.0:
        logger.info('Step 3/5: seed_ratio=1.0，跳过增强')
        aug_ts = timeseries[:n_samples]
        aug_params = params[:n_samples]
    else:
        logger.info(f'Step 3/5: 参数增强（目标 {n_samples} 个）...')
        aug_ts, aug_params = augment_with_parameter_sampling(
            timeseries, params, param_names, aug_cfg, cfg,
            target_n=n_samples, seed=seed + 1,
            max_workers=max_workers if parallel else 1,
            progress_callback=progress_callback,
        )
    logger.info(f'  → 增强后 {aug_ts.shape[0]} 个样本')

    logger.info('Step 4/5: 转换为18维统一参数...')
    converter = PowerSystemParamConverter()
    unified_list = []
    for i in range(len(aug_params)):
        params_dict = {name: float(aug_params[i, j]) for j, name in enumerate(param_names)}
        unified_list.append(converter.convert(scenario_name, params_dict))
    unified_params = np.array(unified_list, dtype=np.float32)
    logger.info(f'  → 参数维度: {len(param_names)}维 → 18维')

    logger.info('Step 5/5: 写入 HDF5...')
    metadata = {
        'config_path': cfg_path,
        'scenario_name': scenario_name,
        'simulator': 'transient',
        'simulation_type': 'transient',
        'n_original': timeseries.shape[0],
        'n_augmented': aug_ts.shape[0],
        'augmentation_method': 'parameter_sampling',
        'param_names': converter.param_names,
        'unified_params_version': 'v1.0',
        'original_param_names': param_names,
    }
    save_dataset(output_path, aug_ts, unified_params, converter.param_names, metadata)

    elapsed = time.time() - t0
    logger.info(f'===== 完成！耗时 {elapsed:.1f}s =====')
    logger.info(f'数据集形状: timeseries={aug_ts.shape}, params={unified_params.shape}')
    logger.info(f'输出文件: {output_path}')
    return output_path


def main():
    parser = argparse.ArgumentParser(description='暂态稳定数据合成管线（ANDES）')
    parser.add_argument('--config', type=str, required=True, help='YAML 配置文件路径')
    parser.add_argument('--n-samples', type=int, default=None, help='目标样本数')
    parser.add_argument('--parallel', action='store_true', help='启用多线程并行')
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'配置文件不存在: {args.config}')

    run_pipeline(args.config, parallel=args.parallel, max_workers=args.max_workers, n_samples=args.n_samples)


if __name__ == '__main__':
    main()
