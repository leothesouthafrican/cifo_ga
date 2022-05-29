from attr import evolve
import numpy as np
import math
import time
start_time = time.time()

from genetics import Individual, Environment, NN_Engine, Population
from crossover import *
from mutation import *
from selection import *



environment_1 = Environment(environment_size=15)

population_1 = Population(size = 200, environment_used = environment_1, optim = "max")

population_1.evolve(
    gens=500,
    select=tournament,
    crossover=pmx_co,
    mutate=inversion_mutation,
    co_p=0.9,
    mu_p=0.1, 
    elitism=True
)

print(population_1.individuals[0].representation)
print(population_1.individuals[-1].representation)
print("*********")
print(population_1.individuals[0].fitness)
print(population_1.individuals[-1].fitness)
print("--- %s seconds ---" % (time.time() - start_time))
print(population_1.informazione_df)
print(population_1.informazione_meta)