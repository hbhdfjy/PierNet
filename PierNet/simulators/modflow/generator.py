"""
MODFLOW 地下水位时序数据生成器。

使用 flopy 构建单层非承压含水层模型，通过参数采样批量生成
地下水位随时间变化的时序数据集。

输入参数（全部为标量，第一梯队）：
  - hk: 水力传导系数 K（m/day）
  - sy: 储水系数（无量纲）
  - pumping: 抽水量 Q（m³/day，负值）
  - strt: 初始水头（m）
  - rch: 面状补给量（m/day）

输出：
  - timeseries: [n_wells, n_timesteps] 各观测井水头时序
  - params: 当前样本的参数字典
"""

import numpy as np
import tempfile
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def resolve_modflow_executable(cfg: Dict[str, Any] | None = None) -> str:
    """解析并定位 MODFLOW-2005 可执行文件。"""
    from pathlib import Path

    cfg = cfg or {}
    checked = []
    candidates = [
        cfg.get('modflow_exe'),
        os.environ.get('PierNet_MODFLOW_EXE'),
        str(Path.home() / '.flopy_bin' / 'mf2005'),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.expanduser(str(candidate))
        checked.append(expanded)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded

    checked_text = ', '.join(checked) if checked else '<none>'
    raise FileNotFoundError(
        '未找到可执行的 MODFLOW-2005 程序。'
        '请在配置中设置 modflow_exe，或设置环境变量 PierNet_MODFLOW_EXE；'
        '也可以先下载 mf2005 到 ~/.flopy_bin/mf2005。'
        f'已检查路径: {checked_text}'
    )


def _get_well_positions(nrow: int, ncol: int, n_wells: int) -> list[tuple[int, int]]:
    """
    根据网格大小动态生成观测井位置。

    策略：均匀分布在网格中，避免边界和中心抽水井。
    """
    positions = []

    if n_wells == 3:
        # 3 井：左上、右上、中下
        positions = [
            (nrow // 4, ncol // 4),
            (nrow // 4, 3 * ncol // 4),
            (3 * nrow // 4, ncol // 2),
        ]
    elif n_wells == 5:
        # 5 井：四角 + 中心偏移
        positions = [
            (nrow // 4, ncol // 4),
            (nrow // 4, 3 * ncol // 4),
            (nrow // 2, ncol // 2),
            (3 * nrow // 4, ncol // 4),
            (3 * nrow // 4, 3 * ncol // 4),
        ]
    elif n_wells == 7:
        # 7 井：六边形 + 中心
        positions = [
            (nrow // 4, ncol // 4),
            (nrow // 4, ncol // 2),
            (nrow // 4, 3 * ncol // 4),
            (nrow // 2, ncol // 2),
            (3 * nrow // 4, ncol // 4),
            (3 * nrow // 4, ncol // 2),
            (3 * nrow // 4, 3 * ncol // 4),
        ]
    elif n_wells == 9:
        # 9 井：3x3 网格
        positions = [
            (nrow // 5, ncol // 5),
            (nrow // 5, ncol // 2),
            (nrow // 5, 4 * ncol // 5),
            (nrow // 2, ncol // 5),
            (nrow // 2, ncol // 2),
            (nrow // 2, 4 * ncol // 5),
            (4 * nrow // 5, ncol // 5),
            (4 * nrow // 5, ncol // 2),
            (4 * nrow // 5, 4 * ncol // 5),
        ]
    else:
        # 默认：随机分布
        for i in range(n_wells):
            r = (i + 1) * nrow // (n_wells + 1)
            c = (i % 3 + 1) * ncol // 4
            positions.append((r, c))

    # 确保所有位置都在范围内
    positions = [(min(r, nrow - 1), min(c, ncol - 1)) for r, c in positions]

    return positions


def _validate_params(params: Dict[str, float]) -> bool:
    """
    检查参数是否物理合理（优化建议2）。

    Returns:
        True if valid, False otherwise
    """
    # 渗透率必须为正
    for key in params:
        if key.startswith("hk") and params[key] <= 0:
            logger.warning(f"参数验证失败: {key}={params[key]} <= 0")
            return False

    # 储水系数必须在 (0, 1)
    sy = params.get("sy", 0.15)
    if not (0 < sy < 1):
        logger.warning(f"参数验证失败: sy={sy} 不在 (0, 1) 范围内")
        return False

    # 抽水量通常为负（抽水），正值为注水
    pumping = params.get("pumping", -200)
    if pumping > 0:
        logger.debug(f"注意: pumping={pumping} > 0 (注水模式)")

    return True


def _sample_params(cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, float]:
    """从配置范围中均匀采样一组标量参数。"""
    p = cfg["params"]
    params = {}

    # 检查是否为多层场景
    nlay = cfg["grid"].get("nlay", 1)

    if nlay > 1:
        # 多层场景：采样各层参数
        for i in range(1, nlay + 1):
            hk_key = f"hk_layer{i}"
            strt_key = f"strt_layer{i}"
            if f"{hk_key}_min" in p:
                params[hk_key] = float(rng.uniform(p[f"{hk_key}_min"], p[f"{hk_key}_max"]))
            if f"{strt_key}_min" in p:
                params[strt_key] = float(rng.uniform(p[f"{strt_key}_min"], p[f"{strt_key}_max"]))

        # 垂向渗透率
        if "vka_min" in p:
            params["vka"] = float(rng.uniform(p["vka_min"], p["vka_max"]))
    else:
        # 单层场景：检查是否为非均质场
        if "hk_mean_log_min" not in p:
            # 标准均质场：采样hk和strt
            params["hk"] = float(rng.uniform(p["hk_min"], p["hk_max"]))
            params["strt"] = float(rng.uniform(p["strt_min"], p["strt_max"]))
        else:
            # 非均质场：只采样strt，hk由随机场生成
            params["strt"] = float(rng.uniform(p["strt_min"], p["strt_max"]))

    # 通用参数
    params["sy"] = float(rng.uniform(p["sy_min"], p["sy_max"]))
    params["pumping"] = float(rng.uniform(p["pumping_min"], p["pumping_max"]))

    # 补给参数：检查是否为季节性场景
    if "rch_wet_season_min" not in p:
        # 非季节性：恒定补给
        params["rch"] = float(rng.uniform(p["rch_min"], p["rch_max"]))
    # 季节性场景的rch在后面单独处理

    # 非均质场参数
    if "hk_mean_log_min" in p:
        params["hk_mean_log"] = float(rng.uniform(p["hk_mean_log_min"], p["hk_mean_log_max"]))
        params["hk_std_log"] = float(rng.uniform(p["hk_std_log_min"], p["hk_std_log_max"]))
        params["hk_correlation_length"] = float(rng.uniform(p["hk_correlation_length_min"], p["hk_correlation_length_max"]))

    # 边界条件参数
    if "river_stage_min" in p:
        params["river_stage"] = float(rng.uniform(p["river_stage_min"], p["river_stage_max"]))
        params["river_cond"] = float(rng.uniform(p["river_cond_min"], p["river_cond_max"]))
    if "lake_stage_min" in p:
        params["lake_stage"] = float(rng.uniform(p["lake_stage_min"], p["lake_stage_max"]))
        params["lake_cond"] = float(rng.uniform(p["lake_cond_min"], p["lake_cond_max"]))

    # 季节性参数
    if "rch_wet_season_min" in p:
        params["rch_wet_season"] = float(rng.uniform(p["rch_wet_season_min"], p["rch_wet_season_max"]))
        params["rch_dry_season"] = float(rng.uniform(p["rch_dry_season_min"], p["rch_dry_season_max"]))
        params["wet_season_duration"] = int(rng.uniform(p.get("wet_season_duration_min", 120), p.get("wet_season_duration_max", 240)))

    return params


def _run_modflow(
    params: Dict[str, float],
    cfg: Dict[str, Any],
    work_dir: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """
    构建并运行 MODFLOW 模型，返回观测井水头时序。

    Args:
        params: 参数字典
        cfg: 配置字典
        work_dir: 工作目录
        rng: 随机数生成器（用于非均质场等需要随机性的场景）

    Returns:
        水头数组，形状 [n_wells, n_timesteps]；运行失败返回 None
    """
    try:
        import flopy
    except ImportError:
        raise ImportError("请先安装 flopy：pip install flopy")

    grid = cfg["grid"]
    nrow = grid["nrow"]
    ncol = grid["ncol"]
    nlay = grid["nlay"]
    delr = grid["delr"]
    delc = grid["delc"]
    top = grid["top"]
    botm = grid["botm"]
    n_timesteps = cfg["n_timesteps"]
    n_wells = cfg["n_wells"]
    well_positions = _get_well_positions(nrow, ncol, n_wells)

    model_name = "modflow_sim"

    # 使用已下载的 MODFLOW 可执行文件
    from pathlib import Path
    exe_path = str(Path.home() / '.flopy_bin' / 'mf2005')

    # --- 构建 MODFLOW-2005 模型 ---
    mf = flopy.modflow.Modflow(
        modelname=model_name,
        exe_name=exe_path,
        model_ws=work_dir,
    )

    # 离散化包（DIS）
    flopy.modflow.ModflowDis(
        mf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        nper=n_timesteps,
        perlen=[1.0] * n_timesteps,   # 每个应力期 1 天
        nstp=[1] * n_timesteps,
        steady=[False] * n_timesteps,
    )

    # 基本包（BAS6）：初始水头
    if nlay > 1:
        # 多层：各层不同初始水头
        strt = np.zeros((nlay, nrow, ncol), dtype=np.float32)
        for i in range(nlay):
            strt_value = params.get(f"strt_layer{i+1}", params.get("strt", 7.0))
            strt[i, :, :] = strt_value
    else:
        strt = np.full((nlay, nrow, ncol), params.get("strt", 7.0), dtype=np.float32)

    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    # 四边设为常水头边界（值=-1）
    ibound[:, 0, :] = -1
    ibound[:, -1, :] = -1
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

    # 层流包（LPF）：支持多层和非均质
    if "hk_mean_log" in params:
        # 非均质场：对数正态随机场（P0-2修复：使用传入的rng）
        hk_mean = 10 ** params["hk_mean_log"]
        hk_std = 10 ** params["hk_std_log"]
        # 使用传入的rng确保可重现性
        if rng is None:
            # 如果未传入rng，使用默认种子（向后兼容）
            rng = np.random.default_rng(42)
        hk_field = rng.lognormal(np.log(hk_mean), hk_std, size=(nrow, ncol))
        flopy.modflow.ModflowLpf(mf, hk=hk_field.astype(np.float32), sy=params["sy"], laytyp=1)

    elif nlay > 1:
        # 多层：各层不同渗透率
        hk = []
        for i in range(nlay):
            hk_value = params.get(f"hk_layer{i+1}", params.get("hk", 10.0))
            hk.append(hk_value)
        vka = params.get("vka", 1.0)
        flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, sy=params["sy"], laytyp=1)

    else:
        # 单层：标准参数
        flopy.modflow.ModflowLpf(mf, hk=params.get("hk", 10.0), sy=params["sy"], laytyp=1)

    # 补给包（RCH）：支持季节性变化
    if "rch_wet_season" in params:
        # 季节性变化：雨季/旱季交替
        rch_wet = params["rch_wet_season"]
        rch_dry = params["rch_dry_season"]
        wet_duration = params.get("wet_season_duration", 180)

        rch_data = {}
        for i in range(n_timesteps):
            day_of_year = i % 365
            if day_of_year < wet_duration:
                rch_data[i] = rch_wet
            else:
                rch_data[i] = rch_dry
    else:
        # 恒定补给
        rch_data = {i: params.get("rch", 0.001) for i in range(n_timesteps)}

    flopy.modflow.ModflowRch(mf, rech=rch_data)

    # 井包（WEL）：抽水井
    if "lake_stage" in params:
        # 湖泊场景：抽水井移到左上角
        pump_row, pump_col = nrow // 4, ncol // 4
    else:
        # 默认：抽水井在中心
        pump_row, pump_col = nrow // 2, ncol // 2

    wel_data = {i: [[0, pump_row, pump_col, params.get("pumping", -200.0)]]
                for i in range(n_timesteps)}
    flopy.modflow.ModflowWel(mf, stress_period_data=wel_data)

    # 边界条件：河流或湖泊
    if "river_stage" in params:
        # 河流边界（左侧边界）
        river_stage = params["river_stage"]
        river_cond = params["river_cond"]
        river_bot = river_stage - 5.0
        riv_data = {}
        for sp in range(n_timesteps):
            riv_cells = []
            for r in range(nrow):
                riv_cells.append([0, r, 0, river_stage, river_cond, river_bot])
            riv_data[sp] = riv_cells
        flopy.modflow.ModflowRiv(mf, stress_period_data=riv_data)

    if "lake_stage" in params:
        # 湖泊边界（中心圆形区域）
        lake_stage = params["lake_stage"]
        lake_cond = params["lake_cond"]
        center_r, center_c = nrow // 2, ncol // 2
        radius = 5
        ghb_data = {}
        for sp in range(n_timesteps):
            lake_cells = []
            for r in range(nrow):
                for c in range(ncol):
                    dist = ((r - center_r)**2 + (c - center_c)**2)**0.5
                    if dist <= radius:
                        lake_cells.append([0, r, c, lake_stage, lake_cond])
            ghb_data[sp] = lake_cells
        flopy.modflow.ModflowGhb(mf, stress_period_data=ghb_data)

    # 输出控制包（OC）：输出所有时间步的水头
    stress_period_data = {}
    for kper in range(n_timesteps):
        stress_period_data[(kper, 0)] = ['save head']
    flopy.modflow.ModflowOc(mf, stress_period_data=stress_period_data, compact=True)

    # PCG 求解器（多层模型需要更宽松的收敛条件）
    if nlay > 1:
        flopy.modflow.ModflowPcg(mf, mxiter=100, iter1=50, hclose=0.01, rclose=10.0)
    else:
        flopy.modflow.ModflowPcg(mf)

    # 写入输入文件并运行
    mf.write_input()
    success, buff = mf.run_model(silent=True, report=True)

    if not success:
        # MODFLOW 日志文件后缀为 .list（不是 .lst）
        log_path = os.path.join(work_dir, f"{model_name}.list")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log_lines = f.readlines()
            error_lines = [line for line in log_lines if "ERROR" in line.upper() or "FAILED" in line.upper()]
            if error_lines:
                logger.debug(f"MODFLOW 运行失败，错误信息:\n{''.join(error_lines[-5:])}")
            else:
                logger.debug(f"MODFLOW 运行失败，日志尾部:\n{''.join(log_lines[-5:])}")
        elif buff:
            # 从 run_model 的 stdout 缓冲里找错误信息
            err_lines = [line for line in buff if line.strip() and ("ERROR" in line.upper() or "FAILED" in line.upper())]
            if err_lines:
                logger.debug(f"MODFLOW 运行失败: {err_lines[-1].strip()}")
            else:
                logger.debug("MODFLOW 运行失败（参数不收敛）")
        else:
            logger.debug("MODFLOW 运行失败（无输出）")
        return None

    # 读取水头输出
    hds_path = os.path.join(work_dir, f"{model_name}.hds")
    if not os.path.exists(hds_path):
        logger.warning("未找到水头输出文件，跳过此样本")
        return None

    hds = flopy.utils.HeadFile(hds_path)
    # MODFLOW dry cell 标记值（通常为 -1e30）
    HDRY = -1e29

    # 提取各应力期水头，形状 [nlay, nrow, ncol]
    head_series = []
    for kstpkper in hds.get_kstpkper():
        head = hds.get_data(kstpkper=kstpkper)  # [nlay, nrow, ncol]
        # 将 dry cell 标记值替换为 NaN
        head = np.where(head < HDRY, np.nan, head)
        if nlay > 1:
            # 多层：取所有层的平均水头（忽略 NaN）
            head_series.append(np.nanmean(head, axis=0))
        else:
            head_series.append(head[0])

    # head_series: list of [nrow, ncol]，长度 n_timesteps
    head_array = np.stack(head_series, axis=0)  # [n_timesteps, nrow, ncol]

    # 检查是否有过多 NaN（dry cell 过多说明模型不稳定）
    nan_ratio = np.isnan(head_array).mean()
    if nan_ratio > 0.3:
        logger.warning(f"水头数组 NaN 比例过高 ({nan_ratio:.1%})，跳过此样本")
        return None

    # 提取观测井位置的水头时序
    well_ts = np.zeros((len(well_positions), n_timesteps), dtype=np.float32)
    for i, (r, c) in enumerate(well_positions):
        ts = head_array[:, r, c].copy()
        # 将 dry cell（NaN）用向前填充替换，保持时序连续性
        if np.isnan(ts).any():
            # 向前填充：用上一个有效值填充 NaN
            for t in range(1, len(ts)):
                if np.isnan(ts[t]):
                    ts[t] = ts[t - 1]
            # 如果开头就是 NaN，用第一个有效值向后填充
            if np.isnan(ts[0]):
                first_valid = ts[~np.isnan(ts)]
                if len(first_valid) > 0:
                    ts[np.isnan(ts)] = first_valid[0]
                else:
                    ts[:] = 0.0  # 全部干涸，填 0
        well_ts[i] = ts

    return well_ts  # [n_wells, n_timesteps]


def generate_sample(
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, Dict[str, float]] | tuple[None, None]:
    """
    生成单个样本：采样参数 → 运行 MODFLOW → 返回时序。

    Returns:
        (timeseries [n_wells, n_timesteps], params_dict) 或 (None, None) 若失败
    """
    params = _sample_params(cfg, rng)

    # 参数验证（优化建议2）
    if not _validate_params(params):
        logger.warning("参数验证失败，跳过此样本")
        return None, None

    with tempfile.TemporaryDirectory() as work_dir:
        ts = _run_modflow(params, cfg, work_dir, rng)

    if ts is None:
        return None, None

    return ts, params


def _get_param_names_from_config(cfg: Dict[str, Any]) -> list[str]:
    """
    从配置中动态提取所有参数名称。

    支持：基础参数、多层参数、非均质参数、边界参数、季节参数。
    """
    p = cfg["params"]
    param_names = []

    # 检查多层场景
    nlay = cfg["grid"].get("nlay", 1)

    if nlay > 1:
        # 多层场景：各层参数
        for i in range(1, nlay + 1):
            if f"hk_layer{i}_min" in p:
                param_names.append(f"hk_layer{i}")
            if f"strt_layer{i}_min" in p:
                param_names.append(f"strt_layer{i}")
        if "vka_min" in p:
            param_names.append("vka")
    else:
        # 单层场景：基础参数
        if "hk_min" in p:
            param_names.append("hk")
        if "strt_min" in p:
            param_names.append("strt")

    # 通用参数
    if "sy_min" in p:
        param_names.append("sy")
    if "pumping_min" in p:
        param_names.append("pumping")
    if "rch_min" in p:
        param_names.append("rch")

    # 非均质场参数
    if "hk_mean_log_min" in p:
        param_names.extend(["hk_mean_log", "hk_std_log", "hk_correlation_length"])

    # 边界条件参数
    if "river_stage_min" in p:
        param_names.extend(["river_stage", "river_cond"])
    if "lake_stage_min" in p:
        param_names.extend(["lake_stage", "lake_cond"])

    # 季节性参数
    if "rch_wet_season_min" in p:
        param_names.extend(["rch_wet_season", "rch_dry_season", "wet_season_duration"])

    return param_names


def _generate_single_sample_worker(args):
    """
    并行生成的工作函数（用于 multiprocessing）。

    Args:
        args: (cfg, seed_offset)

    Returns:
        (ts, params) or (None, None) if failed
    """
    cfg, seed_offset = args
    rng = np.random.default_rng(42 + seed_offset)
    try:
        return generate_sample(cfg, rng)
    except Exception as e:
        logger.warning(f"样本生成失败: {e}")
        return None, None


def generate_batch(
    cfg: Dict[str, Any],
    n_samples: int,
    seed: int = 42,
    parallel: bool = False,
    max_workers: int = 10,
    progress_callback=None,
    progress_total: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    批量生成 n_samples 个样本。

    Args:
        cfg: 配置字典（来自 modflow.yaml）
        n_samples: 目标样本数
        seed: 随机种子
        parallel: 是否使用并行生成（单场景内并行）
        max_workers: 并行进程数

    Returns:
        timeseries: [N, n_wells, n_timesteps]
        params_array: [N, n_params]
        param_names: 参数名称列表
    """
    if parallel:
        return _generate_batch_parallel(cfg, n_samples, seed, max_workers,
                                        progress_callback=progress_callback,
                                        progress_total=progress_total or n_samples)
    else:
        return _generate_batch_sequential(cfg, n_samples, seed,
                                          progress_callback=progress_callback,
                                          progress_total=progress_total or n_samples)


def _generate_batch_sequential(
    cfg: Dict[str, Any],
    n_samples: int,
    seed: int = 42,
    progress_callback=None,
    progress_total: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """串行生成（原实现）"""
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    # 动态提取参数名称
    param_names = _get_param_names_from_config(cfg)
    logger.info(f"检测到 {len(param_names)} 个参数: {param_names}")

    ts_list = []
    params_list = []
    attempts = 0
    max_attempts = n_samples * 3  # 允许最多 3 倍失败重试

    import time
    start_time = time.time()

    with tqdm(total=n_samples, desc="生成 MODFLOW 样本",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
        while len(ts_list) < n_samples and attempts < max_attempts:
            ts, params = generate_sample(cfg, rng)
            attempts += 1
            if ts is None:
                continue
            row = [params.get(k, float('nan')) for k in param_names]
            if any(np.isnan(v) for v in row):
                continue
            ts_list.append(ts)
            params_list.append(row)

            # 计算详细进度信息
            success_rate = len(ts_list) / attempts * 100
            elapsed = time.time() - start_time
            samples_per_sec = len(ts_list) / elapsed if elapsed > 0 else 0
            remaining_samples = n_samples - len(ts_list)
            eta_seconds = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0

            # 格式化 ETA
            if eta_seconds > 3600:
                eta_str = f"{eta_seconds/3600:.1f}h"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"

            # 更新进度条
            pbar.set_postfix({
                "成功率": f"{success_rate:.1f}%",
                "尝试": attempts,
                "速度": f"{samples_per_sec:.2f}样本/s",
                "预计剩余": eta_str
            })
            pbar.update(1)
            if progress_callback:
                progress_callback(len(ts_list), progress_total)

    if len(ts_list) == 0:
        raise RuntimeError("所有 MODFLOW 运行均失败，请检查 mf2005 可执行文件是否在 PATH 中")

    timeseries = np.stack(ts_list, axis=0)    # [N, n_wells, n_timesteps]
    params_array = np.array(params_list)       # [N, n_params]

    logger.info(
        f"成功生成 {len(ts_list)}/{n_samples} 个样本（尝试 {attempts} 次，成功率 {len(ts_list)/attempts*100:.1f}%）"
    )
    return timeseries, params_array, param_names


def _generate_batch_parallel(
    cfg: Dict[str, Any],
    n_samples: int,
    seed: int = 42,
    max_workers: int = 10,
    progress_callback=None,
    progress_total: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    并行生成（单场景内并行）。

    Args:
        cfg: 配置字典
        n_samples: 目标样本数
        seed: 随机种子
        max_workers: 并行进程数

    Returns:
        timeseries: [N, n_wells, n_timesteps]
        params_array: [N, n_params]
        param_names: 参数名称列表
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import time

    # 动态提取参数名称
    param_names = _get_param_names_from_config(cfg)
    logger.info(f"检测到 {len(param_names)} 个参数: {param_names}")
    logger.info(f"🚀 并行生成模式：{max_workers} 个进程")

    ts_list = []
    params_list = []
    attempts = 0
    max_attempts = n_samples * 3
    start_time = time.time()

    # 按批次提交，避免一次性提交全部任务导致 executor.__exit__ 等待所有完成
    batch_size = max_workers * 2
    seed_offset = seed

    with tqdm(total=n_samples, desc=f"生成 MODFLOW 样本 ({max_workers}核)",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while len(ts_list) < n_samples and attempts < max_attempts:
                this_batch = min(batch_size, max_attempts - attempts)
                batch_tasks = [(cfg, seed_offset + i) for i in range(this_batch)]
                seed_offset += this_batch
                attempts += this_batch

                futures = [executor.submit(_generate_single_sample_worker, t) for t in batch_tasks]
                for future in as_completed(futures):
                    if len(ts_list) >= n_samples:
                        break
                    try:
                        ts, params = future.result()
                        if ts is not None and params is not None:
                            row = [params.get(k, float('nan')) for k in param_names]
                            if not any(np.isnan(v) for v in row):
                                ts_list.append(ts)
                                params_list.append(row)
                            elapsed = max(time.time() - start_time, 1e-6)
                            rate = len(ts_list) / elapsed
                            pbar.set_postfix({
                                "成功率": f"{len(ts_list)/attempts*100:.1f}%",
                                "速度": f"{rate:.2f}样本/s",
                            })
                            pbar.update(1)
                            logger.info(f"生成进度：{len(ts_list)}/{n_samples}")
                            if progress_callback:
                                progress_callback(len(ts_list), progress_total or n_samples)
                    except Exception as e:
                        logger.warning(f"任务异常: {e}")
                if len(ts_list) >= n_samples:
                    break

    if len(ts_list) == 0:
        raise RuntimeError("所有 MODFLOW 运行均失败，请检查 mf2005 可执行文件是否在 PATH 中")

    # 截取到目标样本数
    ts_list = ts_list[:n_samples]
    params_list = params_list[:n_samples]

    timeseries = np.stack(ts_list, axis=0)    # [N, n_wells, n_timesteps]
    params_array = np.array(params_list)       # [N, n_params]

    logger.info(
        f"✅ 并行生成完成：{len(ts_list)}/{n_samples} 个样本（尝试 {attempts} 次，成功率 {len(ts_list)/attempts*100:.1f}%）"
    )
    logger.info(f"⏱️  总耗时：{time.time() - start_time:.1f}秒，加速比：~{max_workers}×")

    return timeseries, params_array, param_names
