from setuptools import find_packages, setup

setup(
    name="piern-data-synthesis",
    version="0.1.0",
    description="PiERN ??????????????????",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "flopy>=3.7.0",
        "h5py>=3.8.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "piern-modflow=piern.simulators.modflow.pipeline:main",
            "piern-power=piern.simulators.power_flow.pipeline:main",
            "piern-gcam=piern.simulators.gcam.pipeline:main",
        ],
    },
)
