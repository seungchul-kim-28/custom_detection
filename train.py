import hydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import torch.nn as nn
# from omegaconf import DictConfig

## My Module
from src.data.dataloader import build_dataloaders
from src.models.builder import build_model


@hydra.main(version_base=None, config_path="config",)
def main(config):
    train_loader, val_loader = build_dataloaders(config)
    import ipdb; ipdb.set_trace()
    model = build_model(config)
    for epoch in range(1, int(config.runtime.epoch)+1):
        train_one_epoch(train_loader, model, epoch )
        pass
    print(config)

def train_one_epoch(train_loader: DataLoader, model: nn.Module, epoch: int):
    return 1

def evaluate():
    pass

# def 

if __name__ == "__main__":
    main()
