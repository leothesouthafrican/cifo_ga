#Importing modules
from dis import dis
from pickle import NONE
from random import shuffle, choice, sample, random
from operator import attrgetter
from copy import deepcopy
import numpy as np
import pandas as pd
from random import shuffle, choice, sample, random
from utils import da_informazione_a_conoscenza, df_to_excel, excel_concat
import math

#Creating an environment class that holds all the attributes and methods related to creating the virutal environment that
#individual snakes operate within.

class Environment:
    #Initialization
    def __init__(self,apple_position = None, borders = [], environment_size = 20):
        self.apple_position = apple_position
        self.borders = borders
        self.environment_size = environment_size

        #If no apple position is specified (as it can be for debugging purposes) the randomly initialise it.
        #In the future we would like to improve this by ensuring that the apple position is occupied by the snakes current body
        if self.apple_position is None:
            self.apple_position = list(np.random.randint(low=1, high=(self.environment_size - 1), size=2))
        #Creating all of th borders of our environment, this is used to see if the snake ever hits the border and immediately terminates the individual
        #and computes its fitness 
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
    
    #Used to print out most pertinent environment information, mainly during debugging
    def __str__(self):
        return f"Apple Position: {self.apple_position})\nEnvironment Size: {self.environment_size}"

#Create an individual class that holds all the attributes and methods related to each individual snake.
class Individual:
    def __init__(
        self,
        current_environment,
        matrix_weights_1 = None,
        matrix_weights_2 = None,
        matrix_weights_3 = None,
        bias_vector_1 = None,
        bias_vector_2 = None,
        bias_vector_3 = None,
        representation = None,
        snake_head_coordinates = None,
        heading = "N",
        occupied_blocks = None,
        relative_position = [],
        available_epochs = 2000,
        fitness = None,
        fitness_function = "fitness_function_1",
    ):

        #If the matrices used for the NN (the snakes brain) are not specifically set, randomly initialise them
        if matrix_weights_1 is None:
            self.matrix_weights_1 = np.random.uniform(low=-1, high=1, size=(4,10)).round(3)
        if matrix_weights_2 is None:
            self.matrix_weights_2 = np.random.uniform(low=-1, high=1, size=(10,15)).round(3)
        if matrix_weights_3 is None:
            self.matrix_weights_3 = np.random.uniform(low=-1, high=1, size=(15,3)).round(3)
        if bias_vector_1 is None:
            self.bias_vector_1 = np.random.uniform(low=-1, high=1, size=(1,10)).round(3)
        if bias_vector_2 is None:
            self.bias_vector_2 = np.random.uniform(low=-1, high=1, size=(1,15)).round(3)
        if bias_vector_3 is None:
            self.bias_vector_3 = np.random.uniform(low=-1, high=1, size=(1,3)).round(3)

        #saving all these variables as class attributes
        self.environment = current_environment
        self.snake_head_coordinates = snake_head_coordinates
        self.heading = heading
        self.occupied_blocks = occupied_blocks
        self.relative_position = relative_position
        self.initial_epochs = available_epochs
        self.available_epochs = available_epochs
        self.fitness = fitness
        self.representation = representation
        self.fitness_function = fitness_function

        #Setting the starting position for the snake, if not specifically set
        if self.snake_head_coordinates is None:
            self.snake_head_coordinates = np.random.randint(low=2, high=(self.environment.environment_size - 1), size=2)
            self.occupied_blocks = [list(self.snake_head_coordinates- np.asarray([0,1])),list(self.snake_head_coordinates)]

    #A method that calculates the relative position of the snakes head to the environments walls as well as the sin angle to the apple_position
    #These will then be used as the inputs to the neural network
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

    #Creating the genes of the snake ("representation") that will ultimately be used in crossover etc.
    def create_representation(self):
        if self.representation == None:
            weights_1_vector = np.reshape(self.matrix_weights_1, (1,40))
            weights_2_vector = np.reshape(self.matrix_weights_2, (1,150))
            weights_3_vector = np.reshape(self.matrix_weights_3, (1,45))
            representation = np.hstack([weights_1_vector,self.bias_vector_1,weights_2_vector,self.bias_vector_2,weights_3_vector,self.bias_vector_3])
            representation = representation.tolist()[0]
            self.representation = representation

            return self.representation

    def __str__(self):
        return f"Av. Epoch: {self.available_epochs}\nCurrent relative position: {self.relative_position} \nCurrent Heading: {self.heading} \nCurrent Occupied Blocks: {self.occupied_blocks} \nSnake Fitness: {self.fitness}"

#A class that handles everything related to the Neural Network of each snake or the decision making of the snake 
class NN_Engine:

    #initialisation
    def __init__(self, individual, environment):
        self.individual = individual
        self.environment = environment

    #Softmax method for calculating probability of turning and moving in a specific direction
    def softmax(self,x):
        e_x = np.exp(x)
        return e_x/ e_x.sum()
    
    #Defining a sigmoid function used for the activation of hidden layers
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    #Computing each of the hidden layers and activating them in order
    def compute_layers(self):
        individual = self.individual

        hidden_layer_1 = np.dot(individual.relative_position, individual.matrix_weights_1) + individual.bias_vector_1

        activated_layer_1 = self.sigmoid(hidden_layer_1)

        hidden_layer_2 = np.dot(activated_layer_1,individual.matrix_weights_2) + individual.bias_vector_2

        activated_layer_2 = self.sigmoid(hidden_layer_2)

        pre_output = np.dot(activated_layer_2,individual.matrix_weights_3) + individual.bias_vector_3

        final_output = self.softmax(pre_output)

        return final_output

    #Converting the output with the highest probability to a direction
    def chosen_direction(self):
        output_vector = self.compute_layers()
        max_index = np.argmax(output_vector)

        return max_index

    #Checking to see if when the snake_head is on a new block, if there is an apple there
    def check_for_apple(self):
        environment = self.environment
        individual = self.individual
        if individual.snake_head_coordinates != environment.apple_position:
            return True
    #Checking if the snake_head is on a border block
    def check_for_borders(self, new_head_position):
        environment = self.environment
        for border_block in environment.borders:
            if border_block == new_head_position:
                return True
    #Checking if the snake_head is on its body
    def check_for_occupied_block(self, new_head_position):
        individual = self.individual
        for block in individual.occupied_blocks[:-1]:
            if block == new_head_position:
                return True
    
    #The method that handles the movement of the snake throughout the environment
    def update_individual_epoch(self):

        individual = self.individual
        environment = self.environment
        #Call the method that handles the choosing of the snakes next position
        direction = self.chosen_direction()

        #Based on the current state of the snake and given the new direction that it has chose, update its position accordingly
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

        #Updating the snakes head position
        individual.occupied_blocks.append(new_snake_head)
        individual.snake_head_coordinates = new_snake_head
        #Decreasing the number of total allowed moves that the snake has before it dies of "old age"
        individual.available_epochs -= 1

        #If it does not eat an apple, then pop the last block of its body off
        if self.check_for_apple():
            individual.occupied_blocks = individual.occupied_blocks[1:]
        #If it does eat an apple, then pop then dont pop off the last block and create a new apple
        else:
            new_random_coordinates = np.random.randint(low=1, high=(environment.environment_size - 1), size=2)
            environment.apple_position = list(new_random_coordinates)

        #If the snake eats a border block then terminate the snake by setting its available epochs to 0 and compute fitness
        if self.check_for_borders(individual.snake_head_coordinates):
            self.counter = individual.available_epochs
            #Get fitness
            individual.available_epochs = 0            
            self.get_fitness()
        
        #If the snake eats itself terminate snake and calculate and get fitness
        elif self.check_for_occupied_block(individual.snake_head_coordinates):
            self.counter = individual.available_epochs
            #Get fitness
            individual.available_epochs = 0
            self.get_fitness()

        #If none of the above happens and the snake has zero available moves then just terminate snake and calculate fitness
        elif individual.available_epochs == 0:
            self.counter = 0
            self.get_fitness()
        
        #Now that we have moved the snake in space, time to update the new relative position of its new position so that we can feed it back into the NN_Engine.
        individual.distance_computer()
    
    #The get fitness function where we define the various fitness functions that we tried
    def get_fitness(self):
        individual = self.individual

        #Steps taken by the snake during its life
        steps = individual.initial_epochs - self.counter
        #Number of apples eaten by way of the length of the snakes body
        score = len(individual.occupied_blocks)

        if individual.fitness_function == "fitness_function_2":
            if score <= 4:
                fitness = 150*score 
            elif score <= 6:
                fitness = 250*score 
            elif score <= 8:
                fitness = 500*score 
            else:
                fitness = 1000*score 
            individual.fitness = fitness

        elif individual.fitness_function == "fitness_function_3":
            if score <= 4:
                fitness = 150*score + steps
            elif score <= 6:
                fitness = 250*score + steps
            elif score <= 8:
                fitness = 500*score + steps
            else:
                fitness = 1000*score + steps
            individual.fitness = fitness
        else:
            fitness = 250*score + steps
            individual.fitness = fitness

    def __str__(self):
        return f"Direction Chosen: {self.chosen_direction()} \nNew Snake: {self.individual.occupied_blocks} \nNew Heading: {self.individual.heading}"   

#Population class that handles the initial creation of multiple individuals and then handles the evolution process through the evolve method
class Population:
    def __init__(self, size, optim, environment_used, output_file_name, informazione_df = None, informazione_meta = None, fitness_used = "fitness_function_1"):

        self.environment = environment_used
        self.individuals = []
        self.size = size
        self.optim = optim
        self.fitness_used = fitness_used
        self.output_file_name = output_file_name

        #Storing all of the meta data used in this generation with informazione_meta and storing all of the metrics using the informazione_df
        self.informazione_df = informazione_df
        self.informazione_meta = informazione_meta

        #Creating base df for informazione_df
        column_names = ["best_fitness","best_fitness_representation","best_fit_length","best_fit_steps","average_fitness","phenotypic_variance", "genotypic_variance"]
        self.informazione_df = pd.DataFrame(columns=column_names)

    def create_initial_population(self):   

        #Create a new individual for the specified population size
        for new_individual in range(self.size):
            self.individuals.append(
                Individual(
                    self.environment,
                    fitness_function = self.fitness_used
                )
            )
            #Now that we a new offspring we need to make it play so that we can feed the neural network
            self.individuals[new_individual].distance_computer()
            self.individuals[new_individual].create_representation()

            #defining an engine for each of the new individuals
            engine = NN_Engine(self.individuals[new_individual], self.environment)

            #While each individual has available moves go through the process of moving, eating etc. 
            while self.individuals[new_individual].available_epochs > 0:
                engine.update_individual_epoch()

    #Evolve method that evolves the population given specific parameters
    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism,runs):
        for run in range(runs):
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

                    #create new_offspring_1
                    new_offspring_1 = Individual(self.environment, representation=offspring1, fitness_function= self.fitness_used)
                    #make him play
                    new_offspring_1.distance_computer()
                    engine = NN_Engine(new_offspring_1, self.environment)
                    while new_offspring_1.available_epochs > 0:
                        engine.update_individual_epoch()
                    new_pop.append(new_offspring_1)

                    if len(new_pop) < self.size:
                        #create new_offspring_2
                        new_offspring_2 = Individual(self.environment, representation=offspring2, fitness_function= self.fitness_used)
                        #make him play
                        new_offspring_2.distance_computer()
                        engine = NN_Engine(new_offspring_2, self.environment)
                        while new_offspring_2.available_epochs > 0:
                            engine.update_individual_epoch()

                        new_pop.append(new_offspring_2)
                    
                if elitism == True:
                    if self.optim == "max":
                        least = min(new_pop, key=attrgetter("fitness")).representation
                    elif self.optim == "min":
                        least = max(new_pop, key=attrgetter("fitness")).representation
                    
                    new_pop_representations = []
                    for individual in new_pop:
                        new_pop_representations.append(individual.representation)
                    index_to_drop = new_pop_representations.index(least)
                    new_pop.pop(index_to_drop)
                    new_pop.append(elite)

                self.individuals = new_pop
                
                #Calculating all of the necessary metrics for storage and further
                print(f"Current Generation: {gen}")
                result = da_informazione_a_conoscenza(self.individuals, gens,select, crossover, mutate,co_p,mu_p,elitism,self.individuals[0].fitness_function)

                #Appending new row to df
                self.informazione_df = self.informazione_df.append(result[1], ignore_index=True)

            #Update the meta data dictionary
            self.informazione_meta = result[0]
            #Append the latest generation worth of metrics to the populations dataframe
            self.informazione_df = self.informazione_df.append(result[0], ignore_index=True)

            #Output all of the information to excel 
            df_to_excel(self.informazione_df, self.informazione_meta, run)

            #Reset population
            self.individuals = []
            #Create fresh population
            self.create_initial_population()

        excel_concat(self.informazione_meta, gens, output_file_name=self.output_file_name)