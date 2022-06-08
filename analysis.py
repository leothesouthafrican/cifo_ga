import pandas as pd
import os
import openpyxl
from genetics import Population, Environment

class Analysis:
    def __init__(self, population):
        self.population = population
    #Function that computes average value per epoch across n runs
    def average_value_per_epoch(self, path, metrics, population):
        
        #List of all files in the directory
        file_list = os.listdir(path)
        #Iterate through all excels
        analysis_dict = {}
        for file in file_list:
            print(f"Current File: {file}")
            combined_df = pd.DataFrame()
            for metric in metrics:
                full_column = []
                for generation in range((population.gens)):
                    print(f"Current Generation: {generation}")
                    
                    #Getting number of sheets contained within excel
                    total_metric = 0
                    for run in range(population.runs):
                        print(f"Current Sheet: {run}")
                        temp_sheet = pd.read_excel(path + file, sheet_name= "Run_" + str(run)) 
                        print(f"Generation: {generation}")
                        value_to_add = int(temp_sheet.iloc[generation,metric])
                        print(f"Value Added: {value_to_add}")
                        total_metric += value_to_add
                    average_metric = total_metric/population.runs
                    print(f"Average Metric: {average_metric}")
                    full_column.append(average_metric)
                print(f"Full Column: {full_column}")
                combined_df[metric] = full_column
                print(combined_df)
            combined_df.to_excel(path + "averaged_" + file)
            print(analysis_dict)



if __name__ == "__main__":
    environment_1 = Environment(environment_size=15)
    population_1 = Population(size = 50, gens = 5,runs = 5,environment_used = environment_1, optim = "max", output_file_name = "test_4", individual_moves=750)
    analysis_1 = Analysis(population_1)
    analysis_1.average_value_per_epoch('/Users/leo/Desktop/nova/optimisation/other/cifo_ga/analysis_test/', [1,3,4,5,6,7], population_1)
