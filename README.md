Net Ecosystem Exchange experiment using machine learning approaches to modeling daily NEE. 
Here can be found the .csv files, containing the environmental variables, used as input for the models training and the .ipynb files with the code to reproduce the experiment.

Daily data that includes the target variable Net Ecosystem Exchange and its predictors used in the experiment made by Bauer et al., (2026) in the paper (under revision) entitled: 
"Modeling daily Net Ecosystem Exchange in a primary forest site in Amazonia using two machine learning approaches". 
The set of predictors contains 20 environmental variables includes local meteorological variables and ecosystem-related variables such as leaf area index and soil temperature e umidity. 

The co-location of MERGE and ERA5 Land datasets was performed by selecting the closest available data point within a 0.25º spatial grid around the K67 coordinates. 
Data integration was carried out after aggregating the ERA5 and MERGE datasets into daily means and cumulative values, aligning them with FLUXNET data availability based on quality control indices (QC = 1 and QC > 0.75). The description of each variable it is available.
