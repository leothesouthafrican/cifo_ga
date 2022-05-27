import numpy as np
import math
from genetics import Individual, Environment, NN_Engine, Population

from random import random
from copy import deepcopy
from operator import attrgetter


environment_1 = Environment(environment_size=8)


population_1 = Population( size = 10, environment_used = environment_1, optim = "max")

for i in population_1.individuals:
    print(i.representation)


snake_1 = Individual(environment_1, snake_head_coordinates = None, heading="N")
engine_1 = NN_Engine(snake_1,environment_1)


snake_1.distance_computer()

print(snake_1.__str__())
print("**********")

while snake_1.available_epochs > 0:
    engine_1.update_individual_epoch()

    print(environment_1.__str__())
    print(snake_1.__str__())
    print("+++++++")
print(snake_1.create_representation())