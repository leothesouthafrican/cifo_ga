import pandas as pd
import os
import openpyxl
from genetics import Population, Environment

class Analysis:
    def __init__(self, input_path, output_path, population):
        self.population = population
        self.input_path = input_path
        self.output_path = output_path
    #Function that computes average value per epoch across n runs
    def average_value_per_epoch(self, metrics, input_path = None,output_path = None, population = None):
        
        population = self.population if population is None else population
        input_path = self.input_path if input_path is None else input_path
        output_path = self.output_path if output_path is None else output_path

        #List of all files in the directory
        file_list = os.listdir(input_path)
        #Iterate through all excels
        analysis_dict = {}
        for file in file_list:
            if file.endswith(".xlsx"):
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
                            temp_sheet = pd.read_excel(input_path + file, sheet_name= "Run_" + str(run)) 
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
                    ["best_fitness","best_fitness_representation","best_fit_length","best_fit_steps","average_fitness","phenotypic_variance", "genotypic_variance"]
                    combined_df.rename(columns={1: "best_fitness", 3: "best_fit_length",4: "best_fit_steps",5: "average_fitness",6: "phenotypic_variance",7: "genotypic_variance"}, inplace=True)
                combined_df.to_excel(output_path + "averaged_" + file)
                print(analysis_dict)