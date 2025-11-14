import argparse

def build_argparser():
    parser = argparse.ArgumentParser(description="OCT training")
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (JSON)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (to resume)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable, e.g. "0,1"')
    parser.add_argument('--seed', default=42, type=int)
    return parser

import torch, json, os

from torch import optim
from torch.utils.tensorboard import SummaryWriter

from src.parse_config import ConfigParser
from src.data_loader.data_loaders import OCTDataLoaders

from src.model import model as model_module
from src.model import loss as loss_module
from src.model.loss import CrossEntropyWeighted
from src.model import metric as metric_module
from src.model.metric import accuracy
from src.model.model import get_model

from src.trainer.trainer import Trainer

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    dl = OCTDataLoaders(**config["data_loader"]["args"])
    train_loader, val_loader, _ = dl.get_loaders()

    # model
    margs = config["arch"]["args"]
    model = get_model(config["arch"]["type"], margs.get("pretrained", True), margs.get("num_classes", 4)).to(device)

    # loss & metrics
    loss = CrossEntropyWeighted(csv_path=config["data_loader"]["args"]["csv_path"])
    metrics = [accuracy]

    # optim
    oargs = config["optimizer"]["args"]
    optimizer = getattr(optim, config["optimizer"]["type"])(model.parameters(), **oargs)

    # logging/checkpoints
    save_dir = os.path.join(config["trainer"]["save_dir"], config["name"])
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "log")) if config["trainer"].get("tensorboard", True) else None

    trainer = Trainer(model, loss, metrics, optimizer, config, train_loader, val_loader,
                      device, lr_scheduler=None, writer=writer, save_dir=save_dir)
    trainer.train(config["trainer"]["epochs"])

if __name__ == "__main__":
    # allow `python train.py -c config.json`
    args = build_argparser()
    cfg = ConfigParser.from_args(args)
    main(cfg.config)