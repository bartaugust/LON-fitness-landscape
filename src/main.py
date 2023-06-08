import hydra
from omegaconf import DictConfig

import logging

from src.CitiesGraph import CitiesGraph
from src.LON import LON


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    cg = CitiesGraph(cfg, cfg.experiment.num_cities)
    for i in range(cfg.experiment.num_experiments):
        lon = LON(cfg, cg)
        lon.find_best_solution()
        print(lon.best_solution_distance)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)
        raise e
