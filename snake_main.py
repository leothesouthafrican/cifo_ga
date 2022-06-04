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
population_1 = Population(size = 50, environment_used = environment_1, optim = "max")

if __name__ == "__main__":
    #Calling evolve method and passing example parameters
    population_1.evolve(gens=200,select=fps,crossover=pmx_co,mutate=inversion_mutation,co_p=0.9,mu_p=0.1,elitism=True, runs=3)




