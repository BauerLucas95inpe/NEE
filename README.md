Net Ecosystem Exchange experiment using machine learning approaches for modeling daily NEE. 
Here you can find the '.csv' files in the "Datasets" folder, containing the environmental variables, used as input for the models training and the '.py' and '.ipynb' files with the code to reproduce the experiment, in the "Experiments" folder.
Besides, you can also see the bash and log files containing the computational resources information and the model's outcomes in the "Bash_Files" and "Logs" folders.

Daily data that includes the target variable Net Ecosystem Exchange and its predictors used in the experiment made by Bauer et al., (2026) in the paper (under revision) entitled: 
"Modeling daily Net Ecosystem Exchange in a primary forest site in Amazonia using machine and deep learning techniques". The set of predictors contains 20 environmental variables which include local meteorological variables and ecosystem-related variables such as leaf area index and soil temperature and humidity. 

The co-location of MERGE and ERA5 Land datasets was performed by selecting the closest available data point within a 0.25º spatial grid around the K67 coordinates. 
Data integration was carried out after aggregating the ERA5 and MERGE datasets into daily means and cumulative values, aligning them with FLUXNET data availability based on quality control indices (QC = 1 and QC > 0.75). The description of each variable is available below:

| Variable | Units | Description | Source |
| --- | --- | --- | --- |
| NEE_VUT_REF | g C m⁻² d⁻¹ | Net Ecosystem Exchange of CO₂ (target variable) | FLUXNET |
| PPFD | µmol photons m⁻² d⁻¹ | Photosynthetic Photon Flux Density | FLUXNET |
| TA | °C | Air Temperature | FLUXNET |
| LE | W m⁻² | Latent heat flux corrected for the thermal capacity of air | FLUXNET |
| H | W m⁻² | Sensible heat flux corrected for the thermal capacity of air | FLUXNET |
| VPD | kPa | Water Vapor Pressure Deficit | FLUXNET |
| SW | W m⁻² | Shortwave Incoming | FLUXNET |
| NETRAD | W m⁻² | Net Radiation | FLUXNET |
| ST(L1-L4) | K | Soil Temperature in layers 1: 0 - 7 cm, 2: 7 - 28 cm, 3: 28 - 100 cm, and 4: 100 - 289 cm. | ERA5 |
| SWV(L1-L4) | m³ m⁻³ | Volumetric Soil Water in layers 1: 0 - 7 cm, 2: 7 - 28 cm, 3: 28 - 100 cm, and 4: 100 - 289 cm. | ERA5 |
| Evavt | m of water equivalent | Evaporation from vegetation transpiration | ERA5 |
| SSRD | J m⁻² | Surface Solar Radiation Downwards | ERA5 |
| STRD | J m⁻² | Surface Thermal Radiation Downwards | ERA5 |
| LAI | m² m⁻² | Leaf Area Index | ERA5 |
| Precipitation | kg m⁻² | Rain | MERGE |
