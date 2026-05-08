import hydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import torch.nn as nn
# from omegaconf import DictConfig

## My Module
from src.data.dataloader import build_dataloaders
from src.models.model_builder import build_model
from src.loss_fn.loss_builder import build_criterion


@hydra.main(version_base=None, config_path="config",)
def main(config):
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config)
    criterion = build_criterion(config)
    import ipdb; ipdb.set_trace()
    for epoch in range(1, int(config.runtime.epoch)+1):
        train_one_epoch(train_loader, model, epoch )
        evaluate(config)

    print(config)

def train_one_epoch(train_loader: DataLoader, model: nn.Module, epoch: int):
    for img, target in train_loader:
        model(img)
    return 1

def evaluate(config):
    pass

# def 

if __name__ == "__main__":
    main()
