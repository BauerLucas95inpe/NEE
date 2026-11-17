# Modeling Daily Net Ecosystem Exchange (NEE) in an Amazonian Primary Forest Site (K67)

[![Status](https://img.shields.io/badge/Paper-Under%20Review-blue)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)]()
[![HPC](https://img.shields.io/badge/HPC-LNCC%20SDumont-orange)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

This repository contains the datasets, source codes, High-Performance Computing (HPC) Slurm submission scripts, execution logs, model predictions, and SHAP interpretability outputs for the study:

> **Bauer, L. O., et al. (2026).** *Modeling daily Net Ecosystem Exchange in a primary forest site in Amazonia using machine and deep learning techniques* (Under Review).

---

## 📋 Overview

Accurate estimation of **Net Ecosystem Exchange (NEE)** in the Amazon rainforest is critical for understanding the global carbon cycle and assessing ecosystem responses to climatic anomalies. Global upscaling products (such as FLUXCOM and Fluxcom-X) often struggle to capture NEE variance in tropical primary forests and tend to overestimate carbon sink capacity.

This project provides a complete, reproducible machine learning (ML) and deep learning (DL) pipeline developed specifically for local-scale daily NEE modeling at the **K67 Tapajós National Forest flux tower site** (Santarém, Pará, Brazil). The framework investigates:
1. **Dual Modeling Paradigms:** Continuous NEE prediction (**regression**) and ecosystem state detection (**binary classification** of carbon sink [$NEE < 0$] vs. source [$NEE > 0$] regimes).
2. **Data Quality vs. Sample Volume Trade-off:** Comparative evaluation of models trained under strict quality control ($QC = 1$) versus relaxed quality thresholds ($QC > 0.75$).
3. **Tabular Deep Learning vs. Classical ML:** Benchmarking tabular deep learning and classical algorithms—including **MLP**, **RF**, **FTT**, **TabPFN** (default and tuned), **SVM**, **CatGB** (CatBoost), **XGB** (XGBoost), **NAM**, and **TabM** (alongside LightGBM)—against the **Fluxcom-X** global baseline.
4. **Model Interpretability (SHAP):** Quantification of physical and biological drivers governing daily carbon fluxes, highlighting the dominant role of deep soil temperature ($30\text{--}100\text{ cm}$) and deep soil moisture ($100\text{--}289\text{ cm}$).

---

## 📂 Repository Structure

All experimental directories (`Code_Experiments`, `LNCC_Request_Resource_Configuration`, and `Results`) are systematically organized into **nine model-specific subdirectories** (`MLP`, `RF`, `FTT`, `TabPFN`, `SVM`, `CatGB`, `XGB`, `NAM`, and `TabM`) to ensure modularity and full reproducibility:

```text
NEE/
├── Datasets/                                 # Processed daily datasets aligned by QC thresholds (NEE_QC_1.csv, NEE_QC_75.csv)
├── Code_Experiments/                         # Training, validation, testing, and SHAP scripts organized by model
│   ├── MLP/                                  # Python scripts (.py)
│   ├── RF/                                   # Python scripts (.py)
│   ├── FTT/                                  # Python scripts (.py)
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

Net Ecosystem Exchange experiment using machine learning and deep learning approaches for modeling daily NEE. 
Here you can find the '.csv' files in the "Datasets" folder, containing the environmental variables, used as input for the models training and the '.py' and '.ipynb' files with the code to reproduce the experiment, in the "Code_Experiments" folder.
Besides, you can also see the bash and log files containing the computational resources information and the model's outcomes in the "LNCC_Request_Resource_Configuration" and "Logs" folders.

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
