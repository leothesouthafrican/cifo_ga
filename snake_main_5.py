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
population_1 = Population(size = 100,gens=150, runs = 10, environment_used = environment_1, individual_moves=750, output_file_name = "test_13")
population_2 = Population(size = 100,gens=150, runs = 10, environment_used = environment_1, individual_moves=750, output_file_name = "test_14")
population_3 = Population(size = 100,gens=150, runs = 10, environment_used = environment_1, individual_moves=750, output_file_name = "test_15")

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




