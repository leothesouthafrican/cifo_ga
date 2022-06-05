#importing modules
import numpy as np
import math
import time
from genetics import Individual, Environment, NN_Engine, Population
from crossover import *
from mutation import *
from selection import *

#Initializing environment
environment_1 = Environment(environment_size=15)

#Initializing population
population_1 = Population(size = 50, environment_used = environment_1, optim = "max", output_file_name = "test_10")
population_2 = Population(size = 50, environment_used = environment_1, optim = "max", output_file_name = "test_11")
population_3 = Population(size = 50, environment_used = environment_1, optim = "max", output_file_name = "test_12")

if __name__ == "__main__":
    #Creating initial population
    population_1.create_initial_population()
    #Calling evolve method and passing example parameters
    population_1.evolve(gens=500,select=fps,crossover=geometric_co,mutate=inversion_mutation,co_p=0.9,mu_p=0.1,elitism=True, runs=10)

    population_2.create_initial_population()
    population_2.evolve(gens=500,select=fps,crossover=geometric_co,mutate=inversion_mutation,co_p=0.9,mu_p=0.1,elitism=False, runs=10)

    population_3.create_initial_population()
    population_3.evolve(gens=500,select=tournament,crossover=geometric_co,mutate=inversion_mutation,co_p=0.9,mu_p=0.1,elitism=False, runs=10)




