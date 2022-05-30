import numpy as np
import math
import time
from multiprocessing import Process
start_time = time.time()

from genetics import Individual, Environment, NN_Engine, Population
from crossover import *
from mutation import *
from selection import *



environment_1 = Environment(environment_size=10)

population_1 = Population(size = 75, environment_used = environment_1, optim = "max")


if __name__ == '__main__':
    p1 = Process(target = population_1.evolve(gens=21,select=tournament,crossover=single_point_co,mutate=swap_mutation,co_p=0.9,mu_p=0.1, elitism=True))
    p2 = Process(target = population_1.evolve(gens=22,select=tournament,crossover=single_point_co,mutate=swap_mutation,co_p=0.9,mu_p=0.1, elitism=True))
    p3 = Process(target = population_1.evolve(gens=23,select=tournament,crossover=single_point_co,mutate=swap_mutation,co_p=0.9,mu_p=0.1, elitism=True))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    
    
    
    
    
