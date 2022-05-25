from dis import dis
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
            self.apple_position = list(np.random.randint(low=1, high=(self.environment_size - 1), size=2))
        
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
        bias_vector_1 = None,
        bias_vector_2 = None,
        snake_head_coordinates = None,
        heading = "N",
        occupied_blocks = None,
        relative_position = [],
        available_epochs = 200,
        fitness = None
    ):

        if matrix_weights_1 is None:
            self.matrix_weights_1 = np.random.rand(4,5)
        if matrix_weights_2 is None:
            self.matrix_weights_2 = np.random.rand(5,3)
        if bias_vector_1 is None:
            self.bias_vector_1 = np.random.rand(1,5)
        if bias_vector_2 is None:
            self.bias_vector_2 = np.random.rand(1,3)

        self.environment = current_environment
        self.snake_head_coordinates = snake_head_coordinates
        self.heading = heading
        self.occupied_blocks = occupied_blocks
        self.relative_position = relative_position
        self.initial_epochs = available_epochs
        self.available_epochs = available_epochs
        self.fitness = fitness

        if self.snake_head_coordinates is None:
            self.snake_head_coordinates = np.random.randint(low=2, high=(self.environment.environment_size - 1), size=2)
            self.occupied_blocks = [list(self.snake_head_coordinates- np.asarray([0,1])),list(self.snake_head_coordinates)]


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
        return f"Av. Epoch: {self.available_epochs}\nCurrent relative position: {self.relative_position} \nCurrent Heading: {self.heading} \nCurrent Occupied Blocks: {self.occupied_blocks} \nSnake Fitness: {self.fitness}"
    
class NN_Engine:

    def __init__(self, individual, environment):
        self.individual = individual
        self.environment = environment

    def softmax(self,x):
        e_x = np.exp(x)
        return e_x/ e_x.sum()
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def compute_layers(self):
        individual = self.individual

        hidden_layer_1 = np.dot(individual.relative_position, individual.matrix_weights_1) #+ individual.bias_vector_1

        activated_layer_1 = self.sigmoid(hidden_layer_1)

        pre_output = np.dot(activated_layer_1,individual.matrix_weights_2) + individual.bias_vector_2

        final_output = self.softmax(pre_output)

        return final_output

    def chosen_direction(self):
        output_vector = self.compute_layers()
        max_index = np.argmax(output_vector)

        return max_index

    def check_for_apple(self):
        environment = self.environment
        individual = self.individual
        if individual.snake_head_coordinates != environment.apple_position:
            return True
    
    def check_for_borders(self, new_head_position):
        environment = self.environment
        for border_block in environment.borders:
            if border_block == new_head_position:
                return True

    def check_for_occupied_block(self, new_head_position):
        individual = self.individual
        for block in individual.occupied_blocks[:-1]:
            if block == new_head_position:
                return True


    def update_individual_epoch(self):
        individual = self.individual
        environment = self.environment
        direction = self.chosen_direction()
        if (direction == 0 and individual.heading == "N") or (direction == 2 and individual.heading == "S") or (direction == 1 and individual.heading == "W"):
            new_snake_head = list(np.asarray(individual.snake_head_coordinates) - np.asarray([1,0]))
            individual.heading = "W"

        elif (direction == 0 and individual.heading == "E") or (direction == 2 and individual.heading == "W") or (direction == 1 and individual.heading == "N"):
            new_snake_head = list(np.asarray(individual.snake_head_coordinates) + np.asarray([0,1]))
            individual.heading = "N"
            
        elif (direction == 0 and individual.heading == "S") or (direction == 2 and individual.heading == "N") or (direction == 1 and individual.heading == "E"):
            new_snake_head = list(np.asarray(individual.snake_head_coordinates) + np.asarray([1,0]))
            individual.heading = "E"
            
        elif (direction == 0 and individual.heading == "W") or (direction == 2 and individual.heading == "E") or (direction == 1 and individual.heading == "S"):
            new_snake_head = list(np.asarray(individual.snake_head_coordinates) - np.asarray([0,1]))
            individual.heading = "S"


        individual.occupied_blocks.append(new_snake_head)
        #print(individual.occupied_blocks)
        individual.snake_head_coordinates = new_snake_head
        
        if self.check_for_borders(individual.snake_head_coordinates):
            self.counter = individual.available_epochs
            #Get fitness
            print("get fitness_border")
            individual.available_epochs = 0
            

        if self.check_for_occupied_block(individual.snake_head_coordinates):
            self.counter = individual.available_epochs
            #Get fitness
            print("get fitness_occupied")
            individual.available_epochs = 0


        #print(f"New snake head: {new_snake_head} \nApple Position: {environment.apple_position}")
        if self.check_for_apple():
            individual.occupied_blocks = individual.occupied_blocks[1:]
        else:
            print("APPLE EATEN!")
            print(environment.apple_position)
            print(individual.snake_head_coordinates)
            new_random_coordinates = np.random.randint(low=1, high=(environment.environment_size - 1), size=2)
            environment.apple_position = list(new_random_coordinates)

        if individual.available_epochs == 0:
            self.counter = 200
            self.get_fitness()


        
        individual.distance_computer()
        individual.available_epochs -= 1

    
    def get_fitness(self):
        print("In function")
        individual = self.individual

        steps = individual.initial_epochs - self.counter
        score = 10

        fitness = ((2**((steps)/10)) * score )+ steps
        print(steps)
        individual.fitness = fitness

    def __str__(self):
        return f"Direction Chosen: {self.chosen_direction()} \nNew Snake: {self.individual.occupied_blocks} \nNew Heading: {self.individual.heading}"   

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

