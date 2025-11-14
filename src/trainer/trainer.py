import time
import torch
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(self, model, criterion, metric_ftns, optimizer, config,
                 train_loader, val_loader, device, lr_scheduler=None, writer=None, save_dir=None):
        self.model = model
        self.criterion = criterion
        self.metrics = metric_ftns
        self.optimizer = optimizer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.save_dir = save_dir
        self.monitor_mode, self.monitor_key = config["trainer"]["monitor"].split()
        self.best = float("inf") if self.monitor_mode == "min" else -float("inf")
        self.early_stop = config["trainer"].get("early_stop", 0)
        self._no_improve = 0

    def _epoch(self, epoch, train=True):
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        epoch_loss, epoch_acc = 0.0, 0.0

        for i, (x, y, _) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            if train:
                self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            if train:
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += self.metrics[0](logits, y)

            if self.writer:
                print(f"Epoch {epoch} [{i}/{len(self.train_loader)}] loss={loss.item():.4f}")

        n = len(loader)
        return epoch_loss / n, epoch_acc / n

    def train(self, epochs):
        if hasattr(self.criterion, "build"):
            self.criterion.build(self.device)

        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._epoch(ep, train=True)
            va_loss, va_acc = self._epoch(ep, train=False)

            if self.writer:
                self.writer.add_scalar("loss/train", tr_loss, ep)
                self.writer.add_scalar("loss/val", va_loss, ep)
                self.writer.add_scalar("acc/train", tr_acc, ep)
                self.writer.add_scalar("acc/val", va_acc, ep)

            key = va_loss if self.monitor_key == "val_loss" else va_acc
            improved = (key < self.best) if self.monitor_mode == "min" else (key > self.best)
            if improved:
                self.best = key
                self._no_improve = 0
                if self.save_dir:
                    torch.save(self.model.state_dict(), f"{self.save_dir}/models/best.pt")
            else:
                self._no_improve += 1

            if self.lr_scheduler:
                self.lr_scheduler.step(va_loss)

            print(f"Epoch {ep:02d} | tr_loss {tr_loss:.4f} tr_acc {tr_acc:.4f} | "
                  f"val_loss {va_loss:.4f} val_acc {va_acc:.4f} | {time.time()-t0:.1f}s")

            if self.early_stop and self._no_improve >= self.early_stop:
                print("Early stopping triggered.")
                break