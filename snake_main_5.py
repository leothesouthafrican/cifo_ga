#importing modules
import numpy as np
import math
import time
from genetics import Individual, Environment, NN_Engine, Population
from analysis import Analysis
from crossover import *
from mutation import *
from selection import *

#Initializing environment
environment_1 = Environment(environment_size=12)

#Initializing population
population_1 = Population(size = 25,gens=5, runs = 3, environment_used = environment_1, optim = "max", output_file_name = "test_13", individual_moves=500)
population_2 = Population(size = 25,gens=5, runs = 3, environment_used = environment_1, optim = "max", output_file_name = "test_14", individual_moves=500)
population_3 = Population(size = 25,gens=5, runs = 3, environment_used = environment_1, optim = "max", output_file_name = "test_15", individual_moves=500)

#Initializing Analysis
analysis_1 = Analysis(input_path = '/Users/leo/Desktop/nova/optimisation/other/cifo_ga/results/',output_path= r'/Users/leo/Desktop/nova/optimisation/other/cifo_ga/analysis_test/', population = population_1)

if __name__ == "__main__":

    population_1.create_initial_population()
    #Calling evolve method and passing example parameters
    population_1.evolve(select=tournament,crossover=single_point_co,mutate=geometric_mutation,co_p=0.9,mu_p=0.1,elitism=True)

    population_2.create_initial_population()
    #Calling evolve method and passing example parameters
    population_2.evolve(select=tournament,crossover=geometric_co,mutate=geometric_mutation,co_p=0.9,mu_p=0.1,elitism=True)

    population_3.create_initial_population()
    #Calling evolve method and passing example parameters
    population_3.evolve(select=tournament,crossover=pmx_co,mutate=geometric_mutation,co_p=0.9,mu_p=0.1,elitism=True)

    #analysis_1.average_value_per_epoch([1,3,4,5,6,7])




