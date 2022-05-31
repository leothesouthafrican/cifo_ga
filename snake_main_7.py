import numpy as np
import math
import time
start_time = time.time()

from genetics import Individual, Environment, NN_Engine, Population
from crossover import *
from mutation import *
from selection import *



environment_1 = Environment(environment_size=10)

population_1 = Population(size = 75, environment_used = environment_1, optim = "max")

population_1.evolve(
	    gens=200,
	    select=tournament,
	    crossover=single_point_co,
	    mutate=geometric_mutation,
	    co_p=0.9,
	    mu_p=0.1, 
	    elitism=True
	)

population_1.evolve(
	    gens=200,
	    select=tournament,
	    crossover=single_point_co,
	    mutate=swap_mutation,
	    co_p=0.9,
	    mu_p=0.1, 
	    elitism=True
	)
