import hydra
# from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="test")
def main(config):
    breakpoint()
    print(config)


if __name__ == "__main__":
    main()
