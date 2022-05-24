import numpy as np
import math
from genetics import Individual, Environment, Population

from random import random
from copy import deepcopy
from operator import attrgetter

def softmax(x):
    e_x = np.exp(x)
    return e_x/ e_x.sum()

environment_1 = Environment(environment_size=15, apple_position = [13,13])
snake_1 = Individual(environment_1, snake_head_coordinates=[10,10], heading="E")

snake_1.distance_computer()

hidden_layer_1 = np.dot(snake_1.relative_position, snake_1.matrix_weights_1)

pre_bias_output = np.dot(hidden_layer_1,snake_1.matrix_weights_2)

bias_added_vector = snake_1.bias_vector + pre_bias_output

final_output = softmax(bias_added_vector)

print(final_output)