# sim-cluster-api
An API that simulates n golf shots giving you a model of your overall shot pattern based on your previous golf shots, 

# Inputs
takes an JSON array length of 123 where the last two values are whether you are right or left handed, and the number of shots simulated, max number is 100,000.  

Input needs to be structured from top-left to right for the algorithm to make sensible estimate of your shot pattern. See golfsimulation.net/program for an example.
