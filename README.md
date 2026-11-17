# Modeling Daily Net Ecosystem Exchange (NEE) in an Amazonian Primary Forest Site (K67)

[![Status](https://img.shields.io/badge/Paper-Under%20Review-blue)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)]()
[![HPC](https://img.shields.io/badge/HPC-LNCC%20SDumont-orange)]()

This repository contains the datasets, source codes, High-Performance Computing (HPC) Slurm submission scripts, execution logs, model estimates, and SHAP interpretability outputs for the study:

> **Bauer, L. O., et al. (2026).** *Modeling daily Net Ecosystem Exchange in a primary forest site in Amazonia using machine and deep learning techniques* (Under Review).

---

## 📋 Abstract & Overview

The Amazon rainforest plays a critical role in the global carbon cycle and Net Ecosystem Exchange (NEE) is crucial for understanding carbon sequestration and the impacts of human-driven disturbances. Initiatives such as FLUXCOM upscale the NEE information globally using machine learning (ML) techniques, but struggle to explain the NEE variance in the Amazon basin and tend to overestimate its sink capacity. In this study, a local data-driven NEE modeling approach in Amazonia was evaluated, as well as the impact of data availability and quality. Data-driven models were developed using standard machine learning methods and deep learning (DL) techniques for tabular data, using two different NEE data quality indices and modelling approaches (regression and classification). By including soil moisture and soil temperature as predictors to better capture ecosystem processes strongly linked to NEE, we establish a new baseline for local NEE modeling in the region. Feature importance analysis using SHAP (Shapley Additive Explanations) identified as the key predictors the soil temperature (28--100 cm), deep-layer soil water (100--289 cm), followed by meteorological variables linked to solar radiation availability and leaf area index (LAI). All models outperformed previous initiatives, capturing seasonal trends with an improvement about 60% for regression metrics (MEF) and 42% sink and source scenarios classification (70--77% accuracy). These findings highlight the potential of ML and DL for regional carbon balance upscaling in the Amazon rainforest and establish a robust basis, incorporating key biological features, for future model developments and NEE estimates.

---

Here you can find the '.csv' files in the "Datasets" folder, containing the environmental variables, used as input for the models training and the '.py' and '.ipynb' files with the code to reproduce the experiment, in the "Code_Experiments" folder.
Besides, you can also see the bash and log files containing the computational resources information and the model's outcomes in the "LNCC_Request_Resource_Configuration" and "Logs" folders.

The experiment utilize daily data that includes the target variable Net Ecosystem Exchange and its predictors used in the experiment made by Bauer et al., (2026) in the paper (under revision) entitled: "Modeling daily Net Ecosystem Exchange in a primary forest site in Amazonia using machine and deep learning techniques". 
The set of predictors contains 20 environmental variables which include local meteorological variables and ecosystem-related variables such as leaf area index and soil temperature and humidity. 

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


## 📂 Repository Structure

All experimental directories (`Code_Experiments`, `LNCC_Request_Resource_Configuration`, and `Results`) are systematically organized into **model-specific subdirectories** (`MLP`, `RF`, `FTT`, `LightGBM`, `TabPFN`, `SVM`, `CatGB`, `XGB`, `NAM`, and `TabM`):

```text
NEE/
├── Datasets/                                 # Processed daily datasets aligned by QC thresholds (NEE_QC_1.csv, NEE_QC_75.csv)
├── Code_Experiments/                         # Training, validation, testing, and SHAP scripts organized by model
│   ├── MLP/                                  # Python scripts (.py)
│   ├── RF/                                   # Python scripts (.py)
│   ├── FTT/                                  # Python scripts (.py)
│   ├── LightGBM/                             # Python scripts (.py)
│   ├── TabPFN/                               # Jupyter Notebooks (.ipynb)
│   ├── SVM/                                  # Python scripts (.py)
│   ├── CatGB/                                # Python scripts (.py)
│   ├── XGB/                                  # Python scripts (.py)
│   ├── NAM/                                  # Python scripts (.py)
│   └── TabM/                                 # Python scripts (.py)
├── LNCC_Request_Resource_Configuration/      # Slurm batch scripts (.sh) for HPC resource allocation at LNCC, organized by model
│   └── [MLP | RF | FTT | LightGBM | SVM | CatGB | XGB | NAM | TabM]/
├── Results/                                  # Experimental outputs, HPC logs, and interpretability analyses
│   ├── SHAP/                                 # SHAP feature importance values plots, organized by model
│   │   └── [MLP | RF | FTT | LightGBM | SVM | TabPFN | CatGB | XGB | NAM | TabM]/
│   ├── Logs/                                 # LNCC Slurm standard output/error log files (.out/.log), organized by model
│   │   └── [MLP | RF | FTT | LightGBM | SVM | TabPFN | CatGB | XGB | NAM | TabM]/
│   └── Models' output/                       # CSV files containing model predictions and evaluation metrics, organized by model
│       └── [MLP | RF | FTT | LightGBM | SVM | TabPFN | CatGB | XGB | NAM | TabM]/
└── README.md                                 # Project documentation and variable specifications
