import math
from pathlib import Path
from typing import Dict, List, Optional

import torch

from logger.visualization import TensorboardWriter
from utils.util import MetricTracker


class BaseTrainer:
    """Base class for all trainers."""

    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        metrics: List,
        optimizer: torch.optim.Optimizer,
        config,
        device: torch.device,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.lr_scheduler = lr_scheduler

        trainer_config = config.config.get('trainer', {})
        self.epochs = trainer_config.get('epochs', 1)
        self.save_period = trainer_config.get('save_period', 1)
        self.monitor = trainer_config.get('monitor', 'off')
        self.early_stop = trainer_config.get('early_stop', math.inf)
        self.log_step = trainer_config.get('log_step', 10)

        self.start_epoch = 1
        self.checkpoint_dir = Path(config.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor_mode, self.monitor_metric = self._init_monitor_mode(self.monitor)
        self.best_metric = -math.inf if self.monitor_mode == 'max' else math.inf
        self.not_improved_count = 0

        self.writer = TensorboardWriter(config.log_dir)
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=self.writer)

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.monitor_mode != 'off':
                try:
                    metric = log.get(self.monitor_metric)
                    if metric is None:
                        raise KeyError
                except KeyError:
                    self.config.get_logger('BaseTrainer').warning(
                        "Warning: Metric '%s' is not found. Disabling monitoring.", self.monitor_metric
                    )
                    self.monitor_mode = 'off'
                    metric = None

                if metric is not None:
                    improved = (metric > self.best_metric) if self.monitor_mode == 'max' else (metric < self.best_metric)
                    if improved:
                        self.best_metric = metric
                        self.not_improved_count = 0
                        self._save_checkpoint(epoch, save_best=True)
                    else:
                        self.not_improved_count += 1
                        if self.not_improved_count > self.early_stop:
                            self.config.get_logger('BaseTrainer').info('Validation performance didn\'t improve for %d epochs', self.early_stop)
                            break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.best_metric,
            'config': self.config.config,
        }

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()

        filename = self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth'
        torch.save(state, filename)
        self.config.get_logger('BaseTrainer').info('Saving checkpoint: %s ...', filename)

        if save_best:
            best_path = self.checkpoint_dir / 'model_best.pth'
            torch.save(state, best_path)
            self.config.get_logger('BaseTrainer').info('Saving current best: %s ...', best_path)

    def _init_monitor_mode(self, monitor: str):
        if monitor == 'off':
            return 'off', ''
        monitor_mode, monitor_metric = monitor.split()
        if monitor_mode not in {'min', 'max'}:
            raise ValueError('Monitor mode must be either min or max')
        return monitor_mode, monitor_metric

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        raise NotImplementedError
