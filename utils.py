
import os

def informazioni_alla_conoscenza(population, gens):
    """Convert all necessary population meta-data into a dictionary that stores
     a pair of the input data, parameters and the generation by generation results.

    Args:
        population (object): _description_
    """
    
    dir_list = os.listdir("./results")
    last_item = dir_list[-1]
    last_index = int(last_item.split("_")[2].split(".")[0])
    
    #Create dictionary
    temp_dictionary = {"meta_data":{"gens":population.evolve.gens}}

    print(f"Dictionary: {temp_dictionary}")

informazioni_alla_conoscenza("object")