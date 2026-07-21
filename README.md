Net Ecosystem Exchange experiment using machine learning approaches to modeling daily NEE. 
Here can be found the .csv files, containing the environmental variables, used as input for the models training and the .py files with the code to reproduce the experiment.
Besides, you can also see the bash and log files containing the computational resources information and the model's outcomes.

Daily data that includes the target variable Net Ecosystem Exchange and its predictors used in the experiment made by Bauer et al., (2026) in the paper (under revision) entitled: 
"Modeling daily Net Ecosystem Exchange in a primary forest site in Amazonia using two machine learning approaches". 
The set of predictors contains 20 environmental variables includes local meteorological variables and ecosystem-related variables such as leaf area index and soil temperature and umidity. 

The co-location of MERGE and ERA5 Land datasets was performed by selecting the closest available data point within a 0.25º spatial grid around the K67 coordinates. 
Data integration was carried out after aggregating the ERA5 and MERGE datasets into daily means and cumulative values, aligning them with FLUXNET data availability based on quality control indices (QC = 1 and QC > 0.75). The description of each variable it is available below:

| Variable | Units | Description | Source |
| NEE_VUT_REF | g C m-2 s-1 | Net Ecosystem Exchange of CO2 (target variable) | FLUXNET |
| PPFD | µmol photons m-2 s-1 | Photosynthetic Photon Flux Density | FLUXNET |
| TA | ºC | Air Temperature | FLUXNET |
| LE | W m-2 | Latent heat flux corrected for the thermal capacity of air | FLUXNET |
| H | W m-2 | Sensible heat flux corrected for the thermal capacity of air | FLUXNET |
| VPD | kPa | Water Vapor Pressure Deficit | FLUXNET |
| SW | W m-2 | Shortwave Incoming | FLUXNET |
| NETRAD | W m-2 | Net Radiation | FLUXNET |
| ST(L1-L4) | K | Soil Temperature in layers 1: 0 - 7 cm, 2: 7 - 28 cm, 3: 28 - 100 cm, and 4: 100 - 289 cm. | ERA5 |
| SWV(L1-L4) | m3m-3 | Volumetric Soil Water in layers 1: 0 - 7 cm, 2: 7 - 28 cm, 3: 28 - 100 cm, and 4: 100 - 289 cm. | ERA5 |
| Evavt | m of water equivalent | Evaporation from vegetation transpiration | ERA5 |
| SSRD | J m-2 | Surface Solar Radiation Downwards | ERA5 |
| STRD | J m-2 | Surface Thermal Radiation Downwards | ERA5 |
| LAI | m2m-2 | Leaf Area Index | ERA5 |
| Precipitation | kg m-2 | Rain | MERGE |
