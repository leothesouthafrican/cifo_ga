import numpy as np
from genetics import Individual

from random import random
from copy import deepcopy
from operator import attrgetter

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


snake_1 = Individual()

snake_1.distance_computer()
print(snake_1.heading)

print(snake_1.relative_position)