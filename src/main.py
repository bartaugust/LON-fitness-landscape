import hydra
from omegaconf import DictConfig

import logging


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    pass


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)
        raise e