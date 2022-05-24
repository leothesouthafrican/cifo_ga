from dis import dis
from random import shuffle, choice, sample, random
from operator import attrgetter
from copy import deepcopy
import numpy as np

class Individual:
    def __init__(
        self,
        matrix_weights = None,
        snake_head_coordinates = [1,1],
        heading = "E",
        occupied_blocks = [(10,10),(10,9)],
        length = None,
        relative_position = []
    ):
        if matrix_weights is None:
            self.matrix_weights = np.random.rand(4,5)

        self.snake_head_coordinates = snake_head_coordinates
        self.heading = heading
        self.occupied_blocks = occupied_blocks
        self.length = len(self.occupied_blocks)
        self.relative_position = relative_position

    def distance_computer(self):

        print("Works!")
        if self.heading == "N":
            print("If Works!")
            distance_left = self.snake_head_coordinates[0]
            distance_right = 19 - self.snake_head_coordinates[0]
            distance_forward = 19 - self.snake_head_coordinates[1]

            self.relative_position = [distance_left,distance_forward,distance_right]
            

        elif self.heading == "E":
            distance_left = 19 - self.snake_head_coordinates[1]
            distance_right = self.snake_head_coordinates[1]
            distance_forward = 19 - self.snake_head_coordinates[0]

            self.relative_position = [distance_left, distance_forward, distance_right]

        elif self.heading == "S":
            distance_left = 19 - self.snake_head_coordinates[0]
            distance_right = self.snake_head_coordinates[0]
            distance_forward = self.snake_head_coordinates[1]

            self.relative_position = [distance_left, distance_forward, distance_right]

        elif self.heading == "W":
            distance_left = self.snake_head_coordinates[1]
            distance_right = 19 - self.snake_head_coordinates[1]
            distance_forward = self.snake_head_coordinates[0]

            self.relative_position = [distance_left, distance_forward, distance_right]

    def __str__(self):
           return f"Current relative position: {self.distance_computer}"

class Population:
    def __init__(self, size, optim, **kwargs):
        self.individuals = []
        self.size = size
        self.optim = optim
        for _ in range(size):
            self.individuals.append(
                Individual(
                    size=kwargs["sol_size"],
                    replacement=kwargs["replacement"],
                    valid_set=kwargs["valid_set"],
                )
            )

    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism):
        for gen in range(gens):
            new_pop = []

            if elitism == True:
                if self.optim == "max":
                    elite = deepcopy(max(self.individuals, key=attrgetter("fitness")))
                elif self.optim == "min":
                    elite = deepcopy(min(self.individuals, key=attrgetter("fitness")))

            while len(new_pop) < self.size:
                parent1, parent2 = select(self), select(self)
                # Crossover
                if random() < co_p:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                # Mutation
                if random() < mu_p:
                    offspring1 = mutate(offspring1)
                if random() < mu_p:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))

            if elitism == True:
                if self.optim == "max":
                    least = min(new_pop, key=attrgetter("fitness"))
                elif self.optim == "min":
                    least = max(new_pop, key=attrgetter("fitness"))
                new_pop.pop(new_pop.index([least]))
                new_pop.append(elite)

            self.individuals = new_pop

            if self.optim == "max":
                print(f'Best Individual: {max(self, key=attrgetter("fitness"))}')
            elif self.optim == "min":
                print(f'Best Individual: {min(self, key=attrgetter("fitness"))}')

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, head_coordinates):
        return self.individuals[head_coordinates]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, individual_size={len(self.individuals[0])})"

