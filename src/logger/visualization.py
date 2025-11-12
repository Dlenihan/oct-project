from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(self, log_dir: Path, enabled: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.enabled = enabled
        self.writer: Optional[SummaryWriter]
        if self.enabled:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None

    def add_scalar(self, tag: str, value, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
