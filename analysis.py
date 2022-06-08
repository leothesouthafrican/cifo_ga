import pandas as pd
import os
import openpyxl
from genetics import Population, Environment

class Analysis:
    def __init__(self, population):
        self.population = population
    #Function that computes average value per epoch across n runs
    def average_value_per_epoch(self, path, metric, population):
        
        #List of all files in the directory
        file_list = os.listdir(path)
        #Iterate through all excels
        for file in file_list:
            print(f"Current File: {file}")
            full_column = []
            for generation in range(population.gens):
                print(f"Current Generation: {generation}")
                #Storing all sheets in each excel file
                wb = openpyxl.load_workbook(path + file)
                #Getting number of sheets contained within excel
                total_metric = 0
                for sheet in wb.sheetnames:
                    print(f"Current Sheet: {sheet}")
                    temp_sheet = pd.read_excel(path + file, sheet_name= sheet) 
                    
                    value_to_add = int(temp_sheet.iloc[generation,metric])
                    print(f"Value Added: {value_to_add}")
                    total_metric += value_to_add
                average_metric = total_metric/len(wb.sheetnames)
                print(f"Average Metric: {average_metric}")
                full_column.append(average_metric)
            print(f"Full Column: {full_column}")
if __name__ == "__main__":
    environment_1 = Environment(environment_size=15)
    population_1 = Population(size = 50, gens = 500,environment_used = environment_1, optim = "max", output_file_name = "test_4", individual_moves=750)
    analysis_1 = Analysis(population_1)
    analysis_1.average_value_per_epoch('/Users/leo/Desktop/nova/optimisation/other/cifo_ga/test_set_1/', 1, population_1)
