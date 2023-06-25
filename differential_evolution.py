import random

import numpy as np


class DifferentialEvolution:
    def __init__(self, pop_size: int, cr: float, f: float):
        self.pop_size = pop_size
        self.cr = cr
        self.f = f

        self.population = []

    def fit(self, fitness, max_iter: int):
        self.dim = len(self.population[0])

        obj_all = [fitness(x) for x in self.population]
        best_vector = self.population[np.argmin(obj_all)]
        best_obj = min(obj_all)
        prev_obj = best_obj

        obj_iter = []
        for _ in range(max_iter):
            for i in range(self.pop_size):
                # Select three random individuals
                indexes = random.sample(
                    [index for index in range(self.pop_size) if index != i], 3
                )

                candidates = [self.population[index] for index in indexes]

                mutated = self.mutation(candidates)
                mutated = self.check_bounds(mutated)

                trial = self.crossover(mutated, self.population[i])

                obj_target = fitness(self.population[i])
                obj_trial = fitness(trial)

                if obj_trial < obj_target:
                    self.population[i] = trial

            obj_all = [
                fitness(x) for x in self.population
            ]  # Update obj_all after modifying the population

            best_obj = min(obj_all)

            if best_obj < prev_obj:
                best_vector = self.population[np.argmin(obj_all)]
                prev_obj = best_obj
                obj_iter.append(best_obj)
                print(
                    "Iteration: %d f([%s]) = %.5f"
                    % (_, np.around(best_vector, decimals=5), best_obj)
                )

        return [best_vector, best_obj, obj_iter]

    def crossover(self, mutated: list, target: list):
        p = np.random.rand(self.dim)

        trial = []
        for i in range(self.dim):
            if p[i] < self.cr:
                trial.append(mutated[i])
            else:
                trial.append(target[i])

        return trial

    def mutation(self, x: list) -> np.ndarray:
        return np.array(x[0]) + np.array(self.f) * (np.array(x[1]) - np.array(x[2]))

    def check_bounds(self, mutated: np.ndarray) -> list:
        return [
            np.clip([mutated[i]], self.bounds[i][0], self.bounds[i][1])[0]
            for i in range(len(self.bounds))
        ]

    def generate_population(self, bounds: list) -> list:
        population = []
        for _ in range(self.pop_size):
            individual = [random.uniform(bound[0], bound[1]) for bound in bounds]
            population.append(individual)

        self.population = population
        self.bounds = bounds
        return population


def sphere_function(x):
    """Sphere function: f(x) = sum(xi^2)"""
    return np.sum(np.square(x))


if __name__ == "__main__":
    de = DifferentialEvolution(pop_size=30, cr=0.9, f=0.5)
    pop = de.generate_population(bounds=[(-5, 5), (-5, 5)])
    x = de.fit(fitness=sphere_function, max_iter=200)

    print(x[0])

    import matplotlib.pyplot as plt

    plt.plot(x[2], ".-")
    plt.xlabel("Improvement number")
    plt.ylabel("Fitness")
    plt.show()
