import hydra
# from omegaconf import DictConfig

## My Module
from src.data.dataloader import build_dataloaders

@hydra.main(version_base=None, config_path="config",)
def main(config):
    import ipdb; ipdb.set_trace()
    a, b = build_dataloaders(config)
    print(config)


if __name__ == "__main__":
    main()
