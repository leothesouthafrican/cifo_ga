import numpy as np
from genetics import Individual

from random import random
from copy import deepcopy
from operator import attrgetter

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


