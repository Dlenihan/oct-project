import abc
from pathlib import Path
from typing import Any, Dict

import torch


class BaseModel(torch.nn.Module, metaclass=abc.ABCMeta):
    """Base class for all models to provide save/load helpers."""

    def __init__(self) -> None:
        super().__init__()

    def save(self, path: Path, optimizer: torch.optim.Optimizer, epoch: int, **kwargs: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        state.update(kwargs)
        torch.save(state, path)

    def load(self, checkpoint_path: Path, map_location: str | torch.device = 'cpu') -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint

    @abc.abstractmethod
    def forward(self, *inputs: Any) -> Any:
        raise NotImplementedError
