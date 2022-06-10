from analysis import Analysis
from snake_main_1 import population_1
analysis_1 = Analysis(input_path = '/Users/leo/Desktop/nova/optimisation/other/cifo_ga/results/',output_path= r'/Users/leo/Desktop/nova/optimisation/other/cifo_ga/analysis_test/', population = population_1)
analysis_1.average_value_per_epoch([1,3,4,5,6,7])