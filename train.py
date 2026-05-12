import hydra
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import torch.nn as nn
# from omegaconf import DictConfig

## My Module
from src.data.dataloader import build_dataloaders
from src.models.model_builder import build_model
from src.loss_fn.loss_builder import build_criterion
from src.loss_fn.matcher import HungarianMatcher

@hydra.main(version_base=None, config_path="config",)
def main(config):
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config)
    criterion = build_criterion(config, matcher=HungarianMatcher(config))
    
    for epoch in range(1, int(config.runtime.epoch)+1):
        train_one_epoch(train_loader, model, epoch, criterion)
        evaluate(config)

    print(config)

def train_one_epoch(train_loader, model, epoch, criterion):
    for img, target in train_loader:
        out = model(img)
        import ipdb; ipdb.set_trace()
        loss = criterion(out, target)
        
    return None

def evaluate(config):
    pass

# def 

if __name__ == "__main__":
    main()
