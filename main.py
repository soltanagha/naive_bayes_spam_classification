from model import *

# Taking .csv file from current directory
ds = loadDataSet()

# Preproccess dataset and calculate probabilities for training ds. 
parameters, probabilities, testSet = trainProbabilities(ds)

# Trying predict spam messages
evaluateTest(parameters, probabilities, testSet)