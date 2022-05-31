from random import randint, sample, uniform


def geometric_mutation(individual, mutation_step = 0.05):
    alpha_list = [None]*len(individual)
    
    for i in range(len(individual)):
        alpha_list[i] = uniform(-mutation_step, mutation_step)
        individual[i] = individual[i] + alpha_list[i]
    
    return individual

def swap_mutation(individual):
    """Swap mutation for a GA individual

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    # Get two mutation points
    mut_points = sample(range(len(individual)), 20)

    for i in range(10):
        # Swap them
        individual[mut_points[i]], individual[mut_points[19-i]] = individual[mut_points[19-i]], individual[mut_points[i]]

    return individual


def inversion_mutation(individual):
    """Inversion mutation for a GA individual

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    # Position of the start and end of substring
    mut_points = sample(range(len(individual)), 2)
    # This method assumes that the second point is after (on the right of) the first one
    # Sort the list
    mut_points.sort()
    # Invert for the mutation
    individual[mut_points[0]:mut_points[1]] = individual[mut_points[0]:mut_points[1]][::-1]

    return individual

