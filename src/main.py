import hydra
from omegaconf import DictConfig

import logging

from src.CitiesGraph import CitiesGraph
from src.LON import LON

import pandas as pd


@hydra.main(version_base='1.3', config_path='../conf', config_name='config')
def main(cfg: DictConfig):
    if type(cfg.experiment.instance) == str:
        tsp_instance = pd.read_csv(cfg.paths.instance_path+cfg.experiment.instance)
        cords = tsp_instance[['x','y']].to_numpy(dtype=int)
        cg = CitiesGraph(cfg, cords_or_num=cords)
    else:
        cg = CitiesGraph(cfg, cords_or_num=cfg.experiment.instance)
    lon = LON(cfg, cg)
    lon.sample_nodes()
    lon.sample_edges()
    lon.calc_metrics()
    lon.save_metrics_to_file()
    print('a')
    # lon.find_best_solution()
    # print(lon.best_solution_distance)
    # lon.visualise_best_solution()
    # lon.plot_solution_distances()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)
        raise e
