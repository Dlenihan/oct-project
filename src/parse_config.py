import argparse
import json
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.util import ensure_dir


def read_json(fname: Path) -> Dict[str, Any]:
    with fname.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(content: Dict[str, Any], fname: Path) -> None:
    with fname.open('w', encoding='utf-8') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class ConfigParser:
    """Utility class that handles configuration parsing and logging setup."""

    def __init__(
        self,
        config: Dict[str, Any],
        resume: Optional[Path] = None,
        modification: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self._config = config
        if modification:
            self._apply_modification(modification)

        self.resume = resume
        self.run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')

        trainer_config = self._config.get('trainer', {})
        self._save_dir = Path(trainer_config.get('save_dir', 'results'))
        exper_name = self._config.get('name', 'experiment')
        self._save_dir = self._save_dir / exper_name / self.run_id
        self._log_dir = self._save_dir / 'log'

        ensure_dir(self._save_dir)
        ensure_dir(self._log_dir)

        # Configure logging as early as possible.
        logging_config = Path(__file__).resolve().parent / 'logger' / 'logger_config.json'
        self.setup_logging(save_dir=self._log_dir, default_path=logging_config)

        self.logger = logging.getLogger('ConfigParser')
        self.logger.info('Save directory set to %s', self._save_dir)
        if resume:
            self.logger.info('Resuming from checkpoint: %s', resume)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ConfigParser':
        config_path = Path(args.config)
        config = read_json(config_path)
        modifications = cls._parse_modification(args.override)
        return cls(config=config, resume=Path(args.resume) if args.resume else None, modification=modifications)

    @staticmethod
    def _parse_modification(cli_options: Optional[str]) -> Optional[Dict[str, Any]]:
        if cli_options is None:
            return None
        modifications: Dict[str, Any] = {}
        for opt in cli_options:
            if '=' not in opt:
                raise ValueError(f'Invalid override option: {opt}. Expected format key=value')
            key, value = opt.split('=', maxsplit=1)
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                parsed_value = value
            modifications[key] = parsed_value
        return modifications

    def _apply_modification(self, modification: Dict[str, Any]) -> None:
        for key_path, value in modification.items():
            keys = key_path.split('.')
            sub_config = self._config
            for key in keys[:-1]:
                if key not in sub_config:
                    sub_config[key] = {}
                sub_config = sub_config[key]
            sub_config[keys[-1]] = value

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)

    def init_obj(self, name: str, module, *args, **kwargs):
        module_config = self._config.get(name, {})
        module_type = module_config.get('type')
        if module_type is None:
            raise KeyError(f"Configuration for '{name}' must contain the field 'type'.")
        module_args = module_config.get('args', {}).copy()
        module_args.update(kwargs)
        return getattr(module, module_type)(*args, **module_args)

    def init_ftn(self, name: str, module, *args, **kwargs):
        module_config = self._config.get(name, {})
        module_type = module_config.get('type')
        if module_type is None:
            raise KeyError(f"Configuration for '{name}' must contain the field 'type'.")
        module_args = module_config.get('args', {}).copy()
        module_args.update(kwargs)
        return getattr(module, module_type)(*args, **module_args)

    def save(self) -> None:
        config_save_path = self.save_dir / 'config.json'
        write_json(self._config, config_save_path)
        self.logger.info('Configuration saved to %s', config_save_path)

    @staticmethod
    def setup_logging(save_dir: Path, default_path: Path, default_level: int = logging.INFO) -> None:
        if default_path.is_file():
            with default_path.open('r', encoding='utf-8') as file:
                config = json.load(file)
            for handler in config.get('handlers', {}).values():
                if 'filename' in handler:
                    handler_path = Path(handler['filename'])
                    handler_path = save_dir / handler_path.name
                    ensure_dir(handler_path.parent)
                    handler['filename'] = str(handler_path)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)


def get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PyTorch OCT Config Parser')
    parser.add_argument('-c', '--config', default='src/config.json', help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, help='Path to the checkpoint to resume from')
    parser.add_argument('-d', '--device', default=None, help='Override device e.g. "cuda:0" or "cpu"')
    parser.add_argument('--override', nargs='*', help='Optional key=value pairs to override config entries')
    return parser


__all__ = ['ConfigParser', 'get_cli_parser', 'read_json', 'write_json']
