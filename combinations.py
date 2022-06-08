import itertools
a = [["fps","tournament"],["geometric","swap","inversion"],["single_point","geometric","pmx","arithmetic"]]
combs = list(itertools.product(*a))
print(combs)

