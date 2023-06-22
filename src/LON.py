import numpy as np
import matplotlib.pyplot as plt

from src.CitiesGraph import CitiesGraph

import logging

from copy import copy


class LON:
    def __init__(self, cfg, cities_graph: CitiesGraph):
        for key, value in cfg.lon.items():
            setattr(self, key, value)

        self.cities_graph = cities_graph

        self.solution = self.generate_initial_solution()
        self.solution_distance = self.cities_graph.calc_total_distance(self.solution)

        self.best_solution = self.solution.copy()
        self.best_solution_distance = self.cities_graph.calc_total_distance(self.best_solution)

        self.best_solution_time = 0
        self.iteration_no = 0

        self.all_solutions = [self.solution.copy()]
        self.all_solution_distances = [self.best_solution_distance]

        self.nodes = []
        self.edges = []
        self.metrics = {}
        self.all_kicks = []

    def generate_initial_solution(self, start=1):
        permutation = np.random.permutation(self.cities_graph.num_cities)
        if start:
            start_idx = np.where(permutation == start)[0][0]
            permutation = np.roll(permutation, -start_idx)
        return permutation

    def generate_new_solution(self, modification='2-opt'):
        new_solution = self.solution.copy()
        if modification == 'swap_random_two':
            indices = np.random.choice(new_solution.size - 1, 2, replace=False) + 1
            new_solution[indices[0]], new_solution[indices[1]] = new_solution[indices[1]], new_solution[indices[0]]
        elif modification == '2-opt':
            indices = np.random.choice(new_solution.size, 2, replace=False)
            beg = new_solution[:indices.min() + 1]
            mid = new_solution[indices.max() - 1:indices.min():-1]
            end = new_solution[indices.max():]
            new_solution = np.concatenate([beg, mid, end])
        else:
            raise Exception('modification: ', modification, ' not found')
        self.compare_solution(new_solution.copy())

    def compare_solution(self, new_solution):
        new_solution_distance = self.cities_graph.calc_total_distance(new_solution)
        self.all_solutions.append(new_solution)
        self.all_solution_distances.append(new_solution_distance)
        if new_solution_distance < self.solution_distance:
            self.solution = new_solution
            self.solution_distance = new_solution_distance
            if self.solution_distance < self.best_solution_distance:
                self.best_solution = self.solution.copy()
                self.best_solution_distance = self.solution_distance
                self.best_solution_time = 0
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

    def plot_solution_distances(self):
        plt.figure()
        plt.plot(self.all_solution_distances)
        plt.show()

    def visualise_best_solution(self):
        self.cities_graph.visualise_route(self.best_solution)

    def apply_random_kick(self, solution):
        new_solution = copy(solution)
        for i in range(self.k):
            indices = np.random.choice(len(new_solution) - 1, 2, replace=False) + 1
            new_solution[indices[0]], new_solution[indices[1]] = new_solution[indices[1]], new_solution[indices[0]]
        return new_solution

    def first_improvement_2opt(self, solution):
        solution1 = copy(solution)
        solution2 = copy(solution)

        is_improved = True
        while is_improved:
            is_improved = False
            for i in range(1, len(solution2) - 1):
                dist = self.cities_graph.calc_total_distance(solution1)
                if is_improved:
                    break
                for j in range(i + 3, len(solution2)):
                    beg = solution2[:i + 1]
                    mid = solution2[j - 1:i:-1]
                    end = solution2[j:]
                    solution2 = np.concatenate([beg, mid, end]).astype(int).tolist()
                    if  dist < self.cities_graph.calc_total_distance(
                            solution2):
                        solution1 = solution2
                        is_improved = True
                        break
                    else:
                        is_improved = False
        return solution1

    def sample_edges(self):
        self.edges = np.zeros((len(self.nodes), len(self.nodes)), dtype=int)
        for node in self.nodes:
            for i in range(self.num_kicks):
                new_solution = self.apply_random_kick(node)
                new_solution = self.first_improvement_2opt(new_solution)
                if new_solution != node:
                    if new_solution in self.nodes:
                        self.all_kicks.append(i)
                        idx1 = self.nodes.index(node)
                        idx2 = self.nodes.index(new_solution)
                        self.edges[idx1, idx2] += 1
                        self.edges[idx2, idx1] += 1
        logging.info('calculated edges')

    def sample_nodes(self):

        for i in range(self.num_nodes):
            for j in range(self.node_attempts):
                self.solution = self.generate_initial_solution()
                self.find_best_solution()
                if self.solution.tolist() not in self.nodes:
                    self.nodes.append(self.solution.tolist())
                    break
        logging.info('calculated nodes')

    def find_sub_sinks(self):
        sub_sinks = [True for i in range(len(self.nodes))]
        for i in range(len(self.nodes)):
            dist_i = self.cities_graph.calc_total_distance(self.nodes[i])
            for j in range(len(self.nodes)):
                if i == j:
                    continue
                if self.edges[i, j] != 0:
                    dist_j = self.cities_graph.calc_total_distance(self.nodes[j])
                    if dist_j < dist_i:
                        sub_sinks[i] = False
                        break

        return sub_sinks

    def calc_con_rel(self):
        sum_con = np.sum(self.edges>0,axis=0)
        sum_non_con = np.sum(self.edges == 0, axis=0)
        return sum_con/sum_non_con
    def calc_metrics(self):
        self.metrics['edgeToNode'] = np.count_nonzero(self.edges) / len(self.nodes)
        self.metrics['escRate'] = np.mean(self.all_kicks)
        self.metrics['numSubSinks'] = sum(self.find_sub_sinks())

        self.metrics['conRel'] = self.calc_con_rel()
