from dis import dis
from msilib.schema import Class
from random import shuffle, choice, sample, random
from operator import attrgetter
from copy import deepcopy
import numpy as np
import math

class Environment:
    def __init__(self,apple_position = None, borders = [], environment_size = 20):
        self.apple_position = apple_position
        self.borders = borders
        self.environment_size = environment_size

        if self.apple_position is None:
            self.apple_position = np.random.randint(low=1, high=self.environment_size, size=2)
        
        if self.borders == []:
            for x in range(0,self.environment_size):
                if x == 0:
                    for y in range(0,self.environment_size):
                        self.borders.append([x,y])
                if x == (self.environment_size - 1) and y != 0:
                    for y in range(0,self.environment_size):
                        self.borders.append([x,y])
                for y in range(0,self.environment_size):
                    if y == (self.environment_size - 1) and x != 0:
                        self.borders.append([x,y])
                    elif y == 0 and x != 0:
                        self.borders.append([x,y])
        
class Individual:
    def __init__(
        self,
        current_environment,
        matrix_weights_1 = None,
        matrix_weights_2 = None,
        bias_vector = None,
        snake_head_coordinates = None,
        heading = "N",
        occupied_blocks = [(10,10),(10,9)],
        length = None,
        relative_position = []
    ):

        if matrix_weights_1 is None:
            self.matrix_weights_1 = np.random.rand(4,5)
        if matrix_weights_2 is None:
            self.matrix_weights_2 = np.random.rand(5,3)
        if bias_vector is None:
            self.bias_vector = np.random.rand(1,3)

        self.environment = current_environment
        self.snake_head_coordinates = snake_head_coordinates
        self.heading = heading
        self.occupied_blocks = occupied_blocks
        self.length = len(self.occupied_blocks)
        self.relative_position = relative_position

        if self.snake_head_coordinates is None:
            self.snake_head_coordinates = np.random.randint(low=1, high=self.environment.environment_size, size=2)


    def distance_computer(self):

        #Loading environment into method for easier access
        environment = self.environment

        #Calculating sin angle from snake_head_coordinates (taking into account heading) to apple_position
        radians = math.atan2(np.abs(self.snake_head_coordinates[1]-environment.apple_position[1]), np.abs(self.snake_head_coordinates[0]- environment.apple_position[0]))
        degrees = math.degrees(radians)

        if self.heading == "N":
            distance_left = self.snake_head_coordinates[0]
            distance_right = (environment.environment_size - 1) - self.snake_head_coordinates[0]
            distance_forward = (environment.environment_size - 1) - self.snake_head_coordinates[1]

            self.relative_position = [distance_left,distance_forward,distance_right, round(np.sin(degrees),2)]
            

        elif self.heading == "E":
            distance_left = (environment.environment_size - 1) - self.snake_head_coordinates[1]
            distance_right = self.snake_head_coordinates[1]
            distance_forward = (environment.environment_size - 1) - self.snake_head_coordinates[0]

            #Adjusting angle for heading
            degrees += 270
            
            self.relative_position = [distance_left, distance_forward, distance_right, round(np.sin(degrees),2)]

        elif self.heading == "S":
            distance_left = (environment.environment_size - 1) - self.snake_head_coordinates[0]
            distance_right = self.snake_head_coordinates[0]
            distance_forward = self.snake_head_coordinates[1]

            #Adjusting angle for heading
            degrees += 180

            self.relative_position = [distance_left, distance_forward, distance_right, round(np.sin(degrees),2)]

        elif self.heading == "W":
            distance_left = self.snake_head_coordinates[1]
            distance_right = (environment.environment_size - 1) - self.snake_head_coordinates[1]
            distance_forward = self.snake_head_coordinates[0]

            #Adjusting angle for heading
            degrees += 90

            self.relative_position = [distance_left, distance_forward, distance_right, round(np.sin(degrees),2)]

    def __str__(self):
           return f"Current relative position: {self.distance_computer}"
    
    class NN_engine:
        def __init__(self, individual):
            self.individual = individual

        def softmax(x):
            e_x = np.exp(x)
            return e_x/ e_x.sum()

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

