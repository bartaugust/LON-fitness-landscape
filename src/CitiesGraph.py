from typing import List
import numpy as np


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
        cities = np.append(cities,cities[0])
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
