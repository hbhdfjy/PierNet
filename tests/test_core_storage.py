import h5py
import numpy as np

from PierNet.core.storage import load_dataset, save_dataset


def test_save_and_load_dataset_roundtrip(tmp_path):
    path = tmp_path / "dataset.h5"
    timeseries = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    params = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    save_dataset(str(path), timeseries, params, ["alpha", "beta"], metadata={"scenario": "case"})

    loaded_ts, loaded_params, names = load_dataset(str(path))

    np.testing.assert_allclose(loaded_ts, timeseries)
    np.testing.assert_allclose(loaded_params, params)
    assert names == ["alpha", "beta"]


def test_load_dataset_accepts_hdf5_string_param_names(tmp_path):
    path = tmp_path / "external.h5"
    with h5py.File(path, "w") as hf:
        hf.create_dataset("timeseries", data=np.zeros((1, 1, 2), dtype=np.float32))
        hf.create_dataset("params", data=np.zeros((1, 2), dtype=np.float32))
        hf.create_dataset(
            "param_names",
            data=np.array(["alpha", "beta"], dtype=h5py.string_dtype(encoding="utf-8")),
        )

    _, _, names = load_dataset(str(path))

    assert names == ["alpha", "beta"]
