from __future__ import annotations


import torch

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser, get_cli_parser
from trainer.trainer import Trainer
from utils.seed import set_seed
from utils.util import prepare_device


def main(config: ConfigParser) -> None:
    logger = config.get_logger('train')
    seed = config.config.get('seed', 42)
    deterministic = config.config.get('deterministic', False)
    set_seed(seed, deterministic=deterministic)
    logger.info('Using seed %d (deterministic=%s)', seed, deterministic)

    n_gpu = config.config.get('n_gpu', 0)
    device, device_ids = prepare_device(n_gpu)
    logger.info('Using device %s with GPUs %s', device, device_ids)

    data_module = config.init_obj('data_loader', module_data)

    model = config.init_obj('model', module_arch)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, met) for met in config.config.get('metrics', [])]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = None
    if 'lr_scheduler' in config.config:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        config=config,
        device=device,
        data_module=data_module,
        lr_scheduler=lr_scheduler,
    )

    if config.resume is not None:
        logger.info('Loading checkpoint from %s', config.resume)
        checkpoint = torch.load(config.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint['state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.start_epoch = checkpoint.get('epoch', 0) + 1
        trainer.best_metric = checkpoint.get('monitor_best', trainer.best_metric)
        if trainer.lr_scheduler is not None and 'lr_scheduler' in checkpoint:
            trainer.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    config.save()
    trainer.train()
    trainer.writer.close()


if __name__ == '__main__':
    parser = get_cli_parser()
    args = parser.parse_args()
    config = ConfigParser.from_args(args)
    main(config)
