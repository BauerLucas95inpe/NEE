# -*- coding: utf-8 -*-
import os

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns
import statistics as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score

# --- MACHINE LEARNING IMPORTS ---
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns

import optuna
from optuna.samplers import TPESampler
import shap

# ==============================================================================
# --- 1. DATA LOADING & PREPARATION ---
# ==============================================================================
print("--- Loading data LightGBM QC75 Experiment---")
Data = pd.read_csv("/scratch/cptec/lucas.bauer/nee_ga_project/data/NEE_QC_75.csv", header=0, decimal='.', sep=",", na_values=-9999)
Data.columns = ["TIMESTAMP", "TA","VPD","PPFD","LE", "H","SW","NETRAD","NEE","SWVL1","STL1","SSRD","STRD","LAI","Evavt","SWVL2","SWVL4","STL2","STL4","STL3","SWVL3","Rain"]

Data['TIMESTAMP'] = pd.to_datetime(Data['TIMESTAMP'])
Data = Data.dropna()

Fold_1 = Data[(Data['TIMESTAMP'] > '2002-01-01') & (Data['TIMESTAMP'] < '2002-06-01')]
Fold_2 = Data[(Data['TIMESTAMP'] > '2002-05-31') & (Data['TIMESTAMP'] < '2002-11-01')]
Fold_3 = Data[(Data['TIMESTAMP'] > '2002-10-31') & (Data['TIMESTAMP'] < '2003-06-01')]
Fold_4 = Data[(Data['TIMESTAMP'] > '2003-05-31') & (Data['TIMESTAMP'] < '2003-11-01')]
Fold_5 = Data[(Data['TIMESTAMP'] > '2003-10-31') & (Data['TIMESTAMP'] < '2004-06-01')]
Fold_6 = Data[(Data['TIMESTAMP'] > '2004-05-31') & (Data['TIMESTAMP'] < '2004-11-01')]
Fold_7 = Data[(Data['TIMESTAMP'] > '2004-10-31') & (Data['TIMESTAMP'] < '2005-06-01')]
Fold_8 = Data[(Data['TIMESTAMP'] > '2005-10-31') & (Data['TIMESTAMP'] < '2006-06-01')]
Fold_9 = Data[(Data['TIMESTAMP'] > '2008-10-31') & (Data['TIMESTAMP'] < '2009-06-01')]
Fold_10 = Data[(Data['TIMESTAMP'] > '2009-05-31') & (Data['TIMESTAMP'] < '2009-11-01')]

# --- NOVOS SPLITS SAZONAIS ---
Wet_Train = pd.concat([Fold_1, Fold_3, Fold_5])
Wet_Val = pd.concat([Fold_7, Fold_8, Fold_9])

Dry_Train = pd.concat([Fold_2, Fold_4, Fold_6])
Dry_Val = Fold_10


# Lista de tuplas contendo: (Nome_do_Cenario, Dados_de_Treino, Dados_de_Validacao)
seasonal_scenarios = [
    ('Estacao_Chuvosa', Wet_Train, Wet_Val),
    ('Estacao_Seca', Dry_Train, Dry_Val)
]

def MEF(y, y_pred):
    median_obs = st.median(y)
    numerator = np.sum((y_pred - y) ** 2)
    denominator = np.sum((y - median_obs) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

GLOBAL_SEED = 23
np.random.seed(GLOBAL_SEED)
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

print(f"--- Reproducibility configured with SEED={GLOBAL_SEED} ---")

# ==============================================================================
# --- 2. OPTUNA HYPERPARAMETER TUNING ---
# ==============================================================================
def objective(trial):
    # Hiperparâmetros otimizados para o LightGBM
    n_estimators = trial.suggest_int('n_estimators', 200, 1000)
    max_depth = trial.suggest_int('max_depth', 4, 32)
    num_leaves = trial.suggest_int('num_leaves', 8, 256) 
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    subsample = trial.suggest_float('subsample', 0.2, 1.0)           
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.2, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 100) 
    min_split_gain = trial.suggest_float('min_split_gain', 0.0, 1.0)
    path_smooth = trial.suggest_float('path_smooth', 0.0, 1.0)
    max_bin = trial.suggest_int('max_bin', 4, 16)

# ==============================================================================
# --- 3. LOOP DE OTIMIZAÇÃO COM OPTUNA E PURGED CV ---
# ==============================================================================
    fold_mse, fold_rmse, fold_r2, fold_mef = [], [], [], []
    list_names = []

    for season_name, train_data, val_data in seasonal_scenarios:
        if train_data.empty or val_data.empty: continue
            
        X_train_f = train_data.drop(columns=['TIMESTAMP', 'NEE'])
        y_train_f = train_data['NEE']
        X_val_f = val_data.drop(columns=['TIMESTAMP', 'NEE'])
        y_val_f = val_data['NEE']

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            random_state=GLOBAL_SEED,
            n_jobs=-1,
            verbosity=-1
        )

        model.fit(X_train_f, y_train_f)
        
        Y_pred_val = model.predict(X_val_f)
        
        mse_val = mean_squared_error(y_val_f, Y_pred_val)
        rmse_val = root_mean_squared_error(y_val_f, Y_pred_val)
        r2_val = r2_score(y_val_f, Y_pred_val)
        mef_val = MEF(y_val_f, Y_pred_val)

        fold_mse.append(mse_val)
        fold_rmse.append(rmse_val)
        fold_r2.append(r2_val)
        fold_mef.append(mef_val)
        list_names.append(season_name)
        
    avg_mse = np.mean(fold_mse)
    
    if trial.number % 50 == 0:
        print(f">> Trial {trial.number} Summary -> "
              f"MSE Mean & SD: {avg_mse:.3f} ({np.std(fold_mse):.3f}) | "
              f"RMSE Mean & SD: {np.mean(fold_rmse):.3f} ({np.std(fold_rmse):.3f}) | "
              f"R2 Mean & SD: {np.mean(fold_r2):.3f} ({np.std(fold_r2):.3f}) | "
              f"MEF Mean & SD: {np.mean(fold_mef):.3f} ({np.std(fold_mef):.3f})")

    trial.set_user_attr("std_mse", np.std(fold_mse))
    trial.set_user_attr("std_rmse", np.std(fold_rmse))
    trial.set_user_attr("std_r2", np.std(fold_r2))
    trial.set_user_attr("std_mef", np.std(fold_mef))
    trial.set_user_attr("mean_r2", np.mean(fold_r2))
    trial.set_user_attr("mean_mef", np.mean(fold_mef))
    trial.set_user_attr("mean_rmse", np.mean(fold_rmse))
    
    trial.set_user_attr("list_names", list_names)
    trial.set_user_attr("list_mse", fold_mse)
    trial.set_user_attr("list_rmse", fold_rmse)
    trial.set_user_attr("list_r2", fold_r2)
    trial.set_user_attr("list_mef", fold_mef)
    
    return avg_mse

print("\n--- Iniciando Optuna Optimization (LightGBM) ---")
optuna.logging.set_verbosity(optuna.logging.WARNING) 
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=98))
study.optimize(objective, n_trials=200)

trials_df_lgb_qc75_cv = study.trials_dataframe()

print("\nBest Trial Params: ")
best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

print("\nBest Trial Metrics (Mean ± SD across Folds):")
print(f"MSE:  {best_trial.value:.4f} ± {best_trial.user_attrs['std_mse']:.4f}")
print(f"RMSE: {best_trial.user_attrs['mean_rmse']:.4f} ± {best_trial.user_attrs['std_rmse']:.4f}")
print(f"R2:   {best_trial.user_attrs['mean_r2']:.4f} ± {best_trial.user_attrs['std_r2']:.4f}")
print(f"MEF:   {best_trial.user_attrs['mean_mef']:.4f} ± {best_trial.user_attrs['std_mef']:.4f}")

print("\nBest Trial Metrics across the folds:")
best_list_mse = best_trial.user_attrs['list_mse']
best_list_rmse = best_trial.user_attrs['list_rmse']
best_list_r2 = best_trial.user_attrs['list_r2']
best_list_mef = best_trial.user_attrs['list_mef']

for i in range(len(best_list_mse)):
    print(f"Fold {i+1:<2} -> MSE: {best_list_mse[i]:.4f} | RMSE: {best_list_rmse[i]:.4f} | R2: {best_list_r2[i]:.4f} | MEF: {best_list_mef[i]:.4f}")
print("=======================================================\n")

# ==============================================================================
# --- 3. FINAL TRAINING VS TEST SET (2010-2011) ---
# ==============================================================================

# 1. Preparação dos DataFrames de Treino Final e Teste
Final_Train_df = pd.concat([Fold_1, Fold_2, Fold_3, Fold_4, Fold_5, Fold_6, Fold_7, Fold_8, Fold_9, Fold_10])
X_train_final = Final_Train_df.drop(columns=['TIMESTAMP', 'NEE'])
y_train_final = Final_Train_df['NEE']

Test_df = Data[Data['TIMESTAMP'] >= '2010-01-01'].copy() 
X_test_final = Test_df.drop(columns=['TIMESTAMP', 'NEE'])
y_test_final = Test_df['NEE']

# 4. Instanciação do Modelo Final com os melhores hiperparâmetros do Optuna
model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            min_split_gain=min_split_gain,
            path_smooth=path_smooth,
            max_bin=max_bin,
            random_state=GLOBAL_SEED,
            n_jobs=-1,
            verbosity=-1
)

print("\n--- Treinando o Modelo Final (LightGBM) ---")
model_final.fit(X_train_final, y_train_final)

Y_pred_test = model_final.predict(X_test_final)

df_test_final = pd.DataFrame({
    'Data': Test_df['TIMESTAMP'],
    'NEE_True': y_test_final,
    'NEE_LightGBM': Y_pred_test
})

df_test_final.to_csv('LightGBM_qc75.csv', index=False)

print(f"R²:    {r2_score(y_test_final, Y_pred_test):.4f}")
print(f"MEF:   {MEF(y_test_final, Y_pred_test):.4f}")
print(f"MSE:   {mean_squared_error(y_test_final, Y_pred_test):.4f}")
print(f"RMSE:  {root_mean_squared_error(y_test_final, Y_pred_test):.4f}")

# ==============================================================================
# --- 3.1. DETAILED XGB PERFORMANCE PLOTS (Scatter & Time Series) ---
# ==============================================================================
print("\n--- Gerando Gráficos Detalhados do LightGBM ---")

Y_true = Test_df['NEE'].values
Y_predicted = Y_pred_test
dates_plot = pd.to_datetime(Test_df['TIMESTAMP']).dt.to_pydatetime()

# 1. Gráfico de Dispersão (Regressão Linear)
slope, intercept = np.polyfit(Y_true, Y_predicted, 1)
equation_text = f'y = {np.round(slope,3)}x + {np.round(intercept,3)}'

fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.scatter(Y_true, Y_predicted, color='green', edgecolors=(0.82, 0.12, 0.12), alpha=0.7)
ax1.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], "k--", lw=1.5)
ax1.plot(Y_true, slope*Y_true + intercept, color='green', lw=1.5)

ax1.set_xlabel("Observed values", fontsize=12)
ax1.set_ylabel("Estimated values using LightGBM", fontsize=12)
ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax1.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.4)
ax1.legend(loc='lower right')
plt.tight_layout()
fig1.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/LightGBM_qc75_dispersion.png', dpi=400)
plt.close(fig1)

# 2. Gráfico de Série Temporal Específico (XGB vs Observado)
fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.plot(dates_plot, Y_true, color='black', marker='o', markersize=6, linestyle='-', linewidth=1, label='Observed values')
ax2.plot(dates_plot, Y_predicted, color='green', marker='^', markersize=6, linestyle=':', linewidth=1.5, label='Estimated values')
ax2.set_ylabel('NEE_VUT_REF (gC m$^{-2}$ d$^{-1}$)', fontsize=12)

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)

ax2.legend(loc='upper center', fontsize=10, bbox_to_anchor=(0.5, 1.15), ncol=2)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.4)
plt.tight_layout()
fig2.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/LightGBM_qc75.png', dpi=400)
plt.close(fig2)

# ==============================================================================
# --- 4. CONTINGENCY TABLE & CLASSIFICATION METRICS ---
# ==============================================================================
print("\n--- Scenario Classification Metrics (LightGBM) ---")

sink_plot_data = []
source_plot_data = []
Y_obs = Test_df['NEE'].values
Y_est = Y_predicted 
model_label = 'LightGBM'
    
# Sink Scenarios (NEE < 0)
a_sink = np.sum((Y_est < 0) & (Y_obs < 0))
b_sink = np.sum((Y_est < 0) & (Y_obs >= 0))
c_sink = np.sum((Y_est >= 0) & (Y_obs < 0))
d_sink = np.sum((Y_est >= 0) & (Y_obs >= 0))
    
total = a_sink + b_sink + c_sink + d_sink
accuracy = (a_sink + d_sink) / total if total > 0 else 0.0
    
POD_sink = a_sink / (a_sink + c_sink) if (a_sink + c_sink) > 0 else 0.0
FAR_sink = b_sink / (a_sink + b_sink) if (a_sink + b_sink) > 0 else 0.0
SR_sink = 1.0 - FAR_sink
BIAS_sink = (a_sink + b_sink) / (a_sink + c_sink) if (a_sink + c_sink) > 0 else 0.0
CSI_sink = a_sink / (a_sink + b_sink + c_sink) if (a_sink + b_sink + c_sink) > 0 else 0.0
    
print(f"Sink Scenarios -> POD: {POD_sink:.3f}, FAR: {FAR_sink:.3f}, BIAS: {BIAS_sink:.3f}, CSI: {CSI_sink:.3f}, Accuracy: {accuracy:.3f}")
sink_plot_data.append({'label': model_label, 'SR': SR_sink, 'POD': POD_sink})

# Source Scenarios (NEE >= 0)
a_source = d_sink
b_source = c_sink
c_source = b_sink
d_source = a_sink

POD_source = a_source / (a_source + c_source) if (a_source + c_source) > 0 else 0.0
FAR_source = b_source / (a_source + b_source) if (a_source + b_source) > 0 else 0.0
SR_source = 1.0 - FAR_source
BIAS_source = (a_source + b_source) / (a_source + c_source) if (a_source + c_source) > 0 else 0.0
CSI_source = a_source / (a_source + b_source + c_source) if (a_source + b_source + c_source) > 0 else 0.0
    
print(f"Source Scenarios -> POD: {POD_source:.3f}, FAR: {FAR_source:.3f}, BIAS: {BIAS_source:.3f}, CSI: {CSI_source:.3f}, Accuracy: {accuracy:.3f}")
source_plot_data.append({'label': model_label, 'SR': SR_source, 'POD': POD_source})

# ==============================================================================
# --- 5. PERFORMANCE DIAGRAM (Roebber Plot) ---
# ==============================================================================
print("\n--- Generating Performance Diagrams (Roebber Plots) ---")

def plot_performance_diagram(plot_data, title, save_path, cbar_label):
    def csi_from_far_pod(far, pod): 
        far = np.clip(far, 0.0001, 0.9999)
        pod = np.clip(pod, 0.0001, 1.0)
        CSI = 1 / (1 / (1 - far) + (1 / pod) - 1)
        return CSI

    far_range, pod_range = np.meshgrid(np.arange(0.0001, 1.005, .005), np.arange(0.0001, 1.005, .005))
    csi_range = csi_from_far_pod(far_range, pod_range)
    thetas = [0.5, 1, 1.5, 2, 4] 

    plt.rc('font', size=10)
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=10)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('legend', fontsize=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    cf = ax.contour(1 - far_range, pod_range, csi_range, levels=np.arange(0, 1.1, 0.1), cmap='coolwarm', zorder=2)
    cf.clabel(fmt='%1.1f')

    for data in plot_data:
        lbl = data['label']
        ax.scatter(data['SR'], data['POD'], c='green', marker='*', edgecolor='black', label=lbl, s=400, zorder=101)

    x_bias = np.linspace(0.001, 1, 100)
    for m in thetas:
        y_bias = m * x_bias
        ax.plot(x_bias, y_bias, '--', c='k', alpha=0.4, zorder=1)

    ay2 = ax.twinx()
    ay2.set_ylim(0, 1)
    ay2.set_yticks([0.5], labels=['0.5'])
    ay2.tick_params(axis='y', length=0)
    
    ax2 = ax.twiny()
    ax2.set_xlim(0, 1)
    ax2.set_xticks([1, 1/1.5, 1/2, 1/4], labels=['1', '1.5', '2', '4'])
    ax2.tick_params(axis='x', length=0)

    sm = mpl.cm.ScalarMappable(cmap='coolwarm')
    sm.set_array([]) 
    cbar = plt.colorbar(sm, ax=ax, label=cbar_label, fraction=0.03, pad=0.05) 

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability of detection - POD')
    ax.set_xlabel('Success rate (1-FAR)')
    ax.set_title(title, pad=25)
    
    legend = ax.legend(frameon=True, loc=2) 
    legend.get_frame().set_alpha(1) 

    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close(fig)

plot_performance_diagram(
    plot_data=sink_plot_data, 
    title='',
    save_path='/scratch/cptec/lucas.bauer/nee_ga_project/data/LightGBM_performance_diagram_sink_qc75.png',
    cbar_label='Critical success index for sink scenarios - CSI'
)
print("Performance Diagram (Sinks) saved!")

plot_performance_diagram(
    plot_data=source_plot_data, 
    title='',
    save_path='/scratch/cptec/lucas.bauer/nee_ga_project/data/LightGBM_performance_diagram_source_qc75.png',
    cbar_label='Critical success index for source scenarios - CSI'
)
print("Performance Diagram (Sources) saved!")

# ==============================================================================
# --- 6. ANÁLISE SHAP (Otimizada para Random Forest) ---
# ==============================================================================
print("\n--- Iniciando Análise SHAP no Conjunto de Treinamento ---")
print("Processando via TreeExplainer")

feature_names = X_train_final.columns.tolist()

explainer_train = shap.TreeExplainer(model_final)
# Passamos os dados originais (sem normalização) para que os valores nos eixos dos gráficos façam sentido prático
shap_values_train = explainer_train.shap_values(X_train_final)

mpl.rc('figure', max_open_warning=0)

print("Salvando gráficos SHAP...")

# Beeswarm
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_train, X_train_final, feature_names=feature_names, show=False, max_display=10)

plt.xlabel(r"SHAP values (Contributions to NEE, gC $m^{-2}$ $d^{-1}$)", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tight_layout()

fig.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/beeswarm_LightGBM_qc75_ss.png', dpi=400, bbox_inches='tight')
plt.close(fig)

# Bar Plot 
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.xlabel(r"Mean (|SHAP value|) (Mean contributions to NEE, gC $m^{-2}$ $d^{-1}$)", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tight_layout()
shap.summary_plot(shap_values_train, X_train_final, feature_names=feature_names, plot_type="bar", show=False, max_display=10)
fig.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/shap_LightGBM_qc75_ss.png', dpi=400, bbox_inches='tight')
plt.close(fig)

print("Experiment LightGBM QC75 Finished")