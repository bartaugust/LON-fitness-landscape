from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import seaborn as sns


class CitiesGraph:
    def __init__(self, cfg, cords_or_num):
        for key, value in cfg.cities_graph.items():
            setattr(self, key, value)

        if type(cords_or_num) == int:
            self.num_cities = cords_or_num
            self.generate_random_cords()
        elif isinstance(cords_or_num, np.ndarray):
            self.cords = cords_or_num
            self.num_cities = len(cords_or_num)
        else:
            raise Exception("cords_or_num must be int or array")

    def generate_random_cords(self):
        self.cords = np.random.randint(self.min_cord, self.max_cord, (self.num_cities, self.dims))

    def calc_distance(self, city1, city2, metric='euclidean'):
        if metric == 'euclidean':
            return np.linalg.norm(self.cords[city1] - self.cords[city2])
        else:
            raise Exception('metric: ', metric, ' not implemented')

    def calc_total_distance(self, cities):
        cities = np.append(cities, cities[0])
        distance = 0
        city2 = -1
        for city in cities:
            city1 = city2
            city2 = city
            if city1 == -1:
                continue
            else:
                distance += self.calc_distance(city1, city2)
        return distance

    def visualise_route(self, cities, show=True):
        plt.figure()
        plt.scatter(self.cords[0, 0], self.cords[0, 1], color='red')
        plt.scatter(self.cords[1:, 0], self.cords[1:, 1])

        cities = np.append(cities, cities[0])
        city2 = -1
        for city in cities:
            city1 = city2
            city2 = city
            if city1 == -1:
                continue
            else:
                plt.plot(self.cords[[city1, city2], 0], self.cords[[city1, city2], 1], 'b', linestyle="--")

        if show:
            plt.show()

    def get_distance_matrix(self, show=True):
        dm = distance_matrix(self.cords, self.cords)
        if show:
            plt.figure()
            sns.heatmap(dm, cmap='hot', annot=True, fmt='.1f', cbar=True)
            plt.show()
        return dm

    def visualise_fitness_landscape(self):
        dm = self.get_distance_matrix(show=False)
