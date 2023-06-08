import numpy as np
from src.CitiesGraph import CitiesGraph

import logging


class LON:
    def __init__(self, cfg, cities_graph: CitiesGraph, start=None):
        for key, value in cfg.lon.items():
            setattr(self, key, value)

        self.cities_graph = cities_graph

        self.solution = self.generate_initial_solution(start=start)
        self.solution_distance = self.cities_graph.calc_total_distance(self.solution)

        self.best_solution = self.solution.copy()
        self.best_solution_distance = self.cities_graph.calc_total_distance(self.best_solution)

        self.best_solution_time = 0
        self.iteration_no = 0

    def generate_initial_solution(self, start=None):
        permutation = np.random.permutation(self.cities_graph.num_cities)
        if start:
            start_idx = np.where(permutation == start)[0][0]
            permutation = np.roll(permutation, -start_idx)
        return permutation

    def generate_new_solution(self, modification='swap_random'):
        new_solution = self.solution.copy()
        if modification == 'swap_random':
            indices = np.random.choice(new_solution.size - 1, 2, replace=False) + 1
            new_solution[indices[0]], new_solution[indices[1]] = new_solution[indices[1]], new_solution[indices[0]]
        else:
            raise Exception('modification: ', modification, ' not found')
        self.compare_solution(new_solution)

    def compare_solution(self, new_solution):
        new_solution_distance = self.cities_graph.calc_total_distance(new_solution)
        if new_solution_distance < self.solution_distance:
            self.solution = new_solution
            self.solution_distance = new_solution_distance
            if self.solution_distance < self.best_solution_distance:
                self.best_solution = self.solution.copy()
                self.best_solution_distance = self.solution_distance
                self.best_solution_time = 0
        else:
            choice = np.random.choice([True, False], size=1, p=[self.p, 1 - self.p])
            if choice[0]:
                self.solution = new_solution
                self.solution_distance = new_solution_distance
        self.max_iter -= 1

    def find_best_solution(self):
        while self.max_iter > 0:
            self.generate_new_solution()
            if self.check_if_better():
                logging.info(f'Early stopping. Solution did not improve in {self.patience} iterations')
                break
        if self.max_iter == 0:
            logging.info('Iteration limit reached')

    def check_if_better(self):
        if self.solution_distance >= self.best_solution_distance:
            self.best_solution_time += 1
            if self.best_solution_time == self.patience:
                return True
        return False
