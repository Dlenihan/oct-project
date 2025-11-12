from __future__ import annotations

from typing import List, Optional

import torch

from base.base_trainer import BaseTrainer
from data_loader.data_loaders import OCTDataModule


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        criterion,
        metrics: List,
        optimizer: torch.optim.Optimizer,
        config,
        device: torch.device,
        data_module: OCTDataModule,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.data_module = data_module
        self.train_loader = data_module.train_loader
        self.valid_loader = data_module.val_loader
        self.test_loader = data_module.test_loader
        self.logger = config.get_logger('Trainer')
        super().__init__(model, criterion, metrics, optimizer, config, device, lr_scheduler)

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (data, target, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            batch_size = data.size(0)
            self.train_metrics.update('loss', loss.item(), n=batch_size)
            for metric in self.metrics:
                metric_value = metric(output.detach(), target.detach())
                self.train_metrics.update(metric.__name__, metric_value, n=batch_size)

            if batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: %d [%d/%d (%.0f%%)] Loss: %.4f',
                    epoch,
                    batch_idx * self.train_loader.batch_size,
                    len(self.train_loader.dataset),
                    100.0 * batch_idx / len(self.train_loader),
                    loss.item(),
                )

        log = self.train_metrics.result()
        if self.valid_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update({f'val_{k}': v for k, v in val_log.items()})
        return log

    def _valid_epoch(self, epoch: int):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for data, target, _ in self.valid_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                batch_size = data.size(0)
                self.valid_metrics.update('loss', loss.item(), n=batch_size)
                for metric in self.metrics:
                    metric_value = metric(output, target)
                    self.valid_metrics.update(metric.__name__, metric_value, n=batch_size)
        val_log = self.valid_metrics.result()
        self.logger.info('Validation Epoch %d: %s', epoch, val_log)
        return val_log
