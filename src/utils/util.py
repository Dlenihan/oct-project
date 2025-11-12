from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


class MetricTracker:
    def __init__(self, *keys: str, writer=None) -> None:
        self.writer = writer
        self._data: Dict[str, Dict[str, float]] = {k: {'total': 0.0, 'count': 0, 'average': 0.0} for k in keys}
        self._global_step = itertools.count()

    def reset(self) -> None:
        for key in self._data:
            self._data[key]['total'] = 0.0
            self._data[key]['count'] = 0
            self._data[key]['average'] = 0.0

    def update(self, key: str, value: float, n: int = 1) -> None:
        if key not in self._data:
            raise KeyError(f'Metric {key} not tracked')
        self._data[key]['total'] += value * n
        self._data[key]['count'] += n
        self._data[key]['average'] = self._data[key]['total'] / self._data[key]['count']

        if self.writer is not None:
            step = next(self._global_step)
            self.writer.add_scalar(key, value, step)

    def avg(self, key: str) -> float:
        if key not in self._data:
            raise KeyError(f'Metric {key} not tracked')
        return self._data[key]['average']

    def result(self) -> Dict[str, float]:
        return {key: value['average'] for key, value in self._data.items()}


def prepare_device(n_gpu_use: int) -> tuple[torch.device, list[int]]:
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        raise ValueError("No GPUs available, but n_gpu_use > 0")
    if n_gpu_use > n_gpu:
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


__all__ = ['ensure_dir', 'MetricTracker', 'prepare_device']
