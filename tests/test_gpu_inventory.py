from piern.training.services import gpu_inventory


def test_gpu_inventory_marks_busy_and_locked(monkeypatch):
    monkeypatch.setattr(gpu_inventory, "query_nvidia_smi", lambda: "0, GPU-A, 10, 12000, 0\n1, GPU-B, 11900, 12000, 0\n2, GPU-C, 10, 12000, 90")

    items = gpu_inventory.build_gpu_inventory(
        jobs=[{"job_id": "train-1", "status": "running", "gpu_id": 0}],
        lock_rows=[],
        active_statuses={"running", "starting", "stopping"},
        free_memory_threshold_mib=2048,
        utilization_threshold=20,
    )

    assert items[0]["available"] is False
    assert items[0]["locked_by_job_id"] == "train-1"
    assert items[1]["reason"] == "memory busy"
    assert items[2]["reason"] == "utilization busy"


def test_gpu_inventory_ignores_queued_jobs(monkeypatch):
    monkeypatch.setattr(gpu_inventory, "query_nvidia_smi", lambda: "0, GPU-A, 10, 12000, 0")

    items = gpu_inventory.build_gpu_inventory(
        jobs=[{"job_id": "train-queued", "status": "queued", "gpu_id": 0}],
        lock_rows=[],
        active_statuses={"queued", "running", "starting", "stopping"},
        free_memory_threshold_mib=2048,
        utilization_threshold=20,
    )

    assert items[0]["available"] is True
    assert items[0]["locked_by_job_id"] is None
