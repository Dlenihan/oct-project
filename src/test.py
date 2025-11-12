from __future__ import annotations

import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser, get_cli_parser
from utils.util import MetricTracker, prepare_device


def evaluate(config: ConfigParser, checkpoint_path: str) -> None:
    logger = config.get_logger('test')
    device, _ = prepare_device(config.config.get('n_gpu', 0))

    data_module = config.init_obj('data_loader', module_data)
    test_loader = data_module.test_loader
    if test_loader is None:
        raise RuntimeError('Test split not available in the CSV file.')

    model = config.init_obj('model', module_arch)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config.config.get('metrics', [])]
    metric_tracker = MetricTracker('loss', *[m.__name__ for m in metrics])
    metric_tracker.reset()

    with torch.no_grad():
        for data, target, _ in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            batch_size = data.size(0)
            metric_tracker.update('loss', loss.item(), n=batch_size)
            for metric in metrics:
                metric_value = metric(output, target)
                metric_tracker.update(metric.__name__, metric_value, n=batch_size)

    results = metric_tracker.result()
    logger.info('Test results: %s', results)
    for key, value in results.items():
        print(f'{key}: {value:.4f}')


if __name__ == '__main__':
    parser = get_cli_parser()
    parser.add_argument('--checkpoint', required=True, help='Path to the checkpoint for evaluation')
    args = parser.parse_args()
    setattr(args, 'resume', args.checkpoint)
    config = ConfigParser.from_args(args)
    evaluate(config, args.checkpoint)
