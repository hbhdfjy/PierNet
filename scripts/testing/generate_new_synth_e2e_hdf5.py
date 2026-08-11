from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small real HDF5 fixture for new-synth E2E tests.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    sample_count = 8
    params = np.asarray(
        [
            [0.8 + 0.1 * index, 10.0 + index, 0.02 + 0.005 * index, 1.5 + 0.2 * index]
            for index in range(sample_count)
        ],
        dtype=np.float32,
    )
    time = np.linspace(0.0, 1.0, 12, dtype=np.float32)
    timeseries = np.empty((sample_count, 3, time.size), dtype=np.float32)
    for index, (stiffness, load, damping, length) in enumerate(params):
        timeseries[index, 0] = load * (1.0 - np.exp(-stiffness * time))
        timeseries[index, 1] = length * np.sin(np.pi * time) * np.exp(-damping * time)
        timeseries[index, 2] = stiffness * load / (1.0 + length * time)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("params", data=params)
        handle.create_dataset("timeseries", data=timeseries)
        handle.create_dataset(
            "param_names",
            data=np.asarray([b"stiffness", b"load", b"damping", b"length"]),
        )


if __name__ == "__main__":
    main()
