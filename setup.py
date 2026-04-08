from setuptools import setup, find_packages

setup(
    name="piern-data-synthesis",
    version="0.1.0",
    description="PiERN 数据合成管线：自动生成高质量训练数据",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # 数值计算
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        # 物理模拟器
        "flopy>=3.7.0",         # MODFLOW 地下水正演
        # 数据存储
        "h5py>=3.8.0",
        # 配置管理
        "pyyaml>=6.0",
        # 进度条
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "piern-modflow=piern.simulators.modflow.pipeline:main",
            "piern-power=piern.simulators.power_system.pipeline:main",
            "piern-gcam=piern.simulators.gcam.pipeline:main",
        ],
    },
)
