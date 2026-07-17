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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score

import optuna
import shap
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. DATA LOADING & PREPARATION ---
# ==============================================================================
print("--- Loading data RF D2 SS Experiment 7 Key Features ---")
Data = pd.read_csv("/scratch/cptec/lucas.bauer/nee_ga_project/data/NEE_QC_75.csv", header=0, decimal='.', sep=",", na_values=-9999)
Data.columns = ["TIMESTAMP", "TA","VPD","PPFD","LE", "H","SW","NETRAD","NEE","SWVL1","STL1","SSRD","STRD","LAI","Evavt","SWVL2","SWVL4","STL2","STL4","STL3","SWVL3","Rain"]

Data['TIMESTAMP'] = pd.to_datetime(Data['TIMESTAMP'])
Data = Data.dropna()

selected_features = ['SWVL4', 'STL3', 'LAI', 'LE', 'STL1', 'SW', 'PPFD']
columns_to_keep = ['TIMESTAMP', 'NEE'] + selected_features
Data = Data[columns_to_keep]

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

Wet_Train = pd.concat([Fold_1, Fold_3, Fold_5])
Wet_Val = pd.concat([Fold_7, Fold_8, Fold_9])

Dry_Train = pd.concat([Fold_2, Fold_4, Fold_6])
Dry_Val = Fold_10

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

# ==============================================================================
# --- 2. OPTUNA HYPERPARAMETER TUNING (Seasonal Split) ---
# ==============================================================================
GLOBAL_SEED = 23
np.random.seed(GLOBAL_SEED)

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 600)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 32)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 16)
    max_features = trial.suggest_int('max_features', low=1, high=7) 
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1)
    min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.2)
    oob_score = trial.suggest_categorical('oob_score', [True, False])
    ccp_alpha = trial.suggest_float('ccp_alpha', 1e-4, 0.1, log=True)

    fold_mse, fold_rmse, fold_r2, fold_mef = [], [], [], []
    list_names = []

    # 2. Evaluate the same configuration in seasonal folds
    for season_name, train_data, val_data in seasonal_scenarios:
        if train_data.empty or val_data.empty: continue
            
        X_train_f = train_data.drop(columns=['TIMESTAMP', 'NEE'])
        y_train_f = train_data['NEE']
        X_val_f = val_data.drop(columns=['TIMESTAMP', 'NEE'])
        y_val_f = val_data['NEE']

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            min_impurity_decrease=min_impurity_decrease,
            oob_score=oob_score,
            ccp_alpha=ccp_alpha,
            bootstrap=True,
            random_state=GLOBAL_SEED,
            n_jobs=-1 
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

print("\n--- Iniciando Optuna Optimization ---")
optuna.logging.set_verbosity(optuna.logging.WARNING) 
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=98))
study.optimize(objective, n_trials=200)

trials_df_rf_cv = study.trials_dataframe()
trials_df_rf_cv.to_csv('/scratch/cptec/lucas.bauer/nee_ga_project/data/7k_mrmr_trials_df_rf_d2_ss.csv', index=False, sep=',')

print("\nBest Trial Params: ")
best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

print("\nBest Trial Metrics (Mean ± SD across Seasons):")
print(f"MSE:  {best_trial.value:.4f} ± {best_trial.user_attrs['std_mse']:.4f}")
print(f"RMSE: {best_trial.user_attrs['mean_rmse']:.4f} ± {best_trial.user_attrs['std_rmse']:.4f}")
print(f"R2:   {best_trial.user_attrs['mean_r2']:.4f} ± {best_trial.user_attrs['std_r2']:.4f}")
print(f"MEF:   {best_trial.user_attrs['mean_mef']:.4f} ± {best_trial.user_attrs['std_mef']:.4f}")

print("\nBest Trial Metrics DETAILED by Season:")
best_list_names = best_trial.user_attrs['list_names']
best_list_mse = best_trial.user_attrs['list_mse']
best_list_rmse = best_trial.user_attrs['list_rmse']
best_list_r2 = best_trial.user_attrs['list_r2']
best_list_mef = best_trial.user_attrs['list_mef']

for i in range(len(best_list_names)):
    print(f"{best_list_names[i]:<16} -> MSE: {best_list_mse[i]:.4f} | RMSE: {best_list_rmse[i]:.4f} | R2: {best_list_r2[i]:.4f} | MEF: {best_list_mef[i]:.4f}")
print("=======================================================\n")


# ==============================================================================
# --- 3. FINAL TRAINING (All 2002-2009 Data) VS TEST SET (2010-2011) ---
# ==============================================================================
def build_best_rf(params):
    return RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
        min_impurity_decrease=params['min_impurity_decrease'],
        oob_score=params['oob_score'],
        ccp_alpha=params['ccp_alpha'],
        bootstrap=True,
        random_state=GLOBAL_SEED,
        n_jobs=-1
    )

Final_Train_df = pd.concat([Wet_Train, Dry_Train, Wet_Val, Dry_Val])
X_train_final = Final_Train_df.drop(columns=['TIMESTAMP', 'NEE'])
y_train_final = Final_Train_df['NEE']

Test_df = Data[Data['TIMESTAMP'] >= '2010-01-01'].copy() 
X_test_final = Test_df.drop(columns=['TIMESTAMP', 'NEE'])
y_test_final = Test_df['NEE']

# Baselines
Test_df['NEE_Persistence'] = Test_df['NEE'].shift(1)
Test_df['NEE_Persistence'] = Test_df['NEE_Persistence'].bfill() 

Final_Train_df['Month'] = Final_Train_df['TIMESTAMP'].dt.month
Final_Train_df['Day'] = Final_Train_df['TIMESTAMP'].dt.day
climatology = Final_Train_df.groupby(['Month', 'Day'])['NEE'].mean().reset_index()

Test_df['Month'] = Test_df['TIMESTAMP'].dt.month
Test_df['Day'] = Test_df['TIMESTAMP'].dt.day
Test_df = Test_df.merge(climatology, on=['Month', 'Day'], how='left', suffixes=('', '_Climatology'))
Test_df.rename(columns={'NEE_Climatology': 'NEE_Seasonal'}, inplace=True)

best_params = best_trial.params
model_final = build_best_rf(best_params)
print("\n--- Treinando o Modelo Random Forest Final ---")
model_final.fit(X_train_final, y_train_final)

Y_pred_test = model_final.predict(X_test_final)

Test_df['NEE_RF'] = Y_pred_test
Test_df_clean = Test_df.dropna(subset=['NEE', 'NEE_RF', 'NEE_Persistence', 'NEE_Seasonal'])

print("\n=======================================================")
print("--- FINAL TEST SET METRICS (RF vs BASELINES) ---")
print("=======================================================")
models_to_evaluate = [
    ('Random Forest', Test_df_clean['NEE_RF']),
    ('Persistence', Test_df_clean['NEE_Persistence']),
    ('Mean Seasonal', Test_df_clean['NEE_Seasonal'])
]

for name, y_pred in models_to_evaluate:
    print(f"\nModel: {name}")
    print(f"R2:   {r2_score(Test_df_clean['NEE'], y_pred):.4f}")
    print(f"MEF:  {MEF(Test_df_clean['NEE'], y_pred):.4f}")
    print(f"MSE:  {mean_squared_error(Test_df_clean['NEE'], y_pred):.4f}")
    print(f"RMSE: {root_mean_squared_error(Test_df_clean['NEE'], y_pred):.4f}")

# ==============================================================================
# --- 3.1. DETAILED RF PERFORMANCE PLOTS (Scatter & Time Series) ---
# ==============================================================================
print("\n--- Gerando Gráficos Detalhados do RF ---")

Y_true = Test_df_clean['NEE'].values
Y_predicted = Test_df_clean['NEE_RF'].values
dates_plot = pd.to_datetime(Test_df_clean['TIMESTAMP']).dt.to_pydatetime()

slope, intercept = np.polyfit(Y_true, Y_predicted, 1)
equation_text = f'y = {np.round(slope,3)}x + {np.round(intercept,3)}'

fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.scatter(Y_true, Y_predicted, color='green', edgecolors=(0.15, 0.7, 0.12), alpha=0.7)
ax1.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], "k--", lw=1.5)
ax1.plot(Y_true, slope*Y_true + intercept, color='green', lw=1.5) 

ax1.set_xlabel("Observed values", fontsize=12)
ax1.set_ylabel("Estimated values using RF", fontsize=12)
ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax1.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.4)
ax1.legend(loc='lower right')
plt.tight_layout()
fig1.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/7k_mrmr_rf_d2_ss_dispersion.png', dpi=400)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.plot(dates_plot, Y_true, color='black', marker='o', markersize=4, linestyle='-', linewidth=1, label='Observed values')
ax2.plot(dates_plot, Y_predicted, color='green', marker='^', markersize=4, linestyle=':', linewidth=1.5, label='Estimated values')
ax2.set_ylabel('NEE_VUT_REF (gC d$^{-1}$)', fontsize=12)

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)

ax2.legend(loc='upper center', fontsize=10, bbox_to_anchor=(0.5, 1.15), ncol=2)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.4)
plt.tight_layout()
fig2.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/7k_mrmr_rf_d2_ss.png', dpi=400)
plt.close(fig2)

# ==============================================================================
# --- 3.2. GENERAL PERFORMANCE PLOT (RF vs Baselines) ---
# ==============================================================================
print("\n--- Gerando Gráfico Comparativo: RF vs Baselines ---")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
fig3, ax3 = plt.subplots(figsize=(12, 8))

x_index = np.arange(len(Test_df_clean))

ax3.plot(x_index, Test_df_clean['NEE'], color='blue', alpha=0.8, linewidth=2, linestyle='-', label='Observed NEE')
ax3.plot(x_index, Test_df_clean['NEE_Seasonal'], color='orange', alpha=0.6, linewidth=1.5, linestyle='-', label='Seasonal Baseline')
ax3.plot(x_index, Test_df_clean['NEE_Persistence'], color='black', alpha=0.6, linewidth=1.5 , linestyle='-', label='Persistence Baseline')
ax3.plot(x_index, Test_df_clean['NEE_RF'], color='green', alpha=0.8, linewidth=2, linestyle='-', label='RF')

ax3.set_ylabel('NEE_VUT_REF (gC m⁻² d⁻¹)', fontweight='normal')
ax3.set_title('', fontsize=18, fontweight='bold', pad=15)

num_ticks = 6
tick_positions = np.linspace(0, len(Test_df_clean) - 1, num_ticks, dtype=int)

tick_labels = [Test_df_clean['TIMESTAMP'].iloc[i].strftime('%Y-%m') for i in tick_positions]

ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=0) 

ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True, edgecolor='black')

sns.despine(bottom=True, left=True)
plt.tight_layout()
fig3.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/7k_mrmr_comparisons_rf_d2_ss.png', dpi=400, bbox_inches='tight')
plt.close(fig3)

# ==============================================================================
# --- 4. CONTINGENCY TABLE & CLASSIFICATION METRICS ---
# ==============================================================================
print("\n--- Scenario Classification Metrics (RF vs Baselines) ---")

sink_plot_data = []
source_plot_data = []
Y_obs = Test_df_clean['NEE'].values

for name, y_est_series in models_to_evaluate:
    Y_est = y_est_series.values
    
    # Sinks (< 0)
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
    
    print(f"\n[{name}] Sink Scenarios -> POD: {POD_sink:.3f}, FAR: {FAR_sink:.3f}, BIAS: {BIAS_sink:.3f}, CSI: {CSI_sink:.3f}, Accuracy: {accuracy:.3f}")
    sink_plot_data.append({'label': name, 'SR': SR_sink, 'POD': POD_sink})

    # Sources (>= 0)
    a_source, b_source, c_source, d_source = d_sink, c_sink, b_sink, a_sink

    POD_source = a_source / (a_source + c_source) if (a_source + c_source) > 0 else 0.0
    FAR_source = b_source / (a_source + b_source) if (a_source + b_source) > 0 else 0.0
    SR_source = 1.0 - FAR_source
    BIAS_source = (a_source + b_source) / (a_source + c_source) if (a_source + c_source) > 0 else 0.0
    CSI_source = a_source / (a_source + b_source + c_source) if (a_source + b_source + c_source) > 0 else 0.0
    
    print(f"[{name}] Source Scenarios -> POD: {POD_source:.3f}, FAR: {FAR_source:.3f}, BIAS: {BIAS_source:.3f}, CSI: {CSI_source:.3f}, Accuracy: {accuracy:.3f}")
    source_plot_data.append({'label': name, 'SR': SR_source, 'POD': POD_source})

# ==============================================================================
# --- 5. PERFORMANCE DIAGRAM (Roebber Plot) ---
# ==============================================================================
print("\n--- Generating Performance Diagrams (Roebber Plots) ---")

def plot_performance_diagram(plot_data, title, save_path, cbar_label):
    def csi_from_far_pod(far, pod): 
        far = np.clip(far, 0.0001, 0.9999)
        pod = np.clip(pod, 0.0001, 1.0)
        return 1 / (1 / (1 - far) + (1 / pod) - 1)

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

    color_map = {'Random Forest': 'red', 'Persistence': 'black', 'Mean Seasonal': 'orange'}
    marker_map = {'Random Forest': '*', 'Persistence': 's', 'Mean Seasonal': 'D'}
    size_map = {'Random Forest': 400, 'Persistence': 120, 'Mean Seasonal': 120}

    for data in plot_data:
        lbl = data['label']
        ax.scatter(data['SR'], data['POD'], c=color_map.get(lbl, 'gray'), marker=marker_map.get(lbl, 'o'), edgecolor='black', label=lbl, s=size_map.get(lbl, 150), zorder=101 if lbl == 'Random Forest' else 100)

    x_bias = np.linspace(0.001, 1, 100)
    for m in thetas:
        ax.plot(x_bias, m * x_bias, '--', c='k', alpha=0.4, zorder=1)

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

plot_performance_diagram(sink_plot_data, '', '/scratch/cptec/lucas.bauer/nee_ga_project/data/7k_mrmr_rf_performance_diagram_sink_d2_ss.png', 'Critical success index for sink scenarios - CSI')
plot_performance_diagram(source_plot_data, '', '/scratch/cptec/lucas.bauer/nee_ga_project/data/7k_mrmr_rf_performance_diagram_source_d2_ss.png', 'Critical success index for source scenarios - CSI')


# ==============================================================================
# --- 6. ANÁLISE SHAP (Otimizada para Random Forest) ---
# ==============================================================================
print("\n--- Iniciando Análise SHAP no Conjunto de Treinamento ---")
print("Processando via TreeExplainer")

feature_names = X_train_final.columns.tolist()

explainer_train = shap.TreeExplainer(model_final)
shap_values_train = explainer_train.shap_values(X_train_final)

mpl.rc('figure', max_open_warning=0)

print("Salvando gráficos SHAP...")

# Beeswarm
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_train, X_train_final, feature_names=feature_names, show=False, max_display=20)
plt.xlabel(r"SHAP values (Contributions to NEE, $\mu mol \ m^{-2} s^{-1}$)", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tight_layout()

fig.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/7k_mrmr_beeswarm_rf_d2_ss.png', dpi=400, bbox_inches='tight')
plt.close(fig)


# Bar Plot 
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.xlabel(r"Mean (|SHAP value|) (Mean contributions to NEE, $\mu mol \ m^{-2} s^{-1}$)", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.tight_layout()
shap.summary_plot(shap_values_train, X_train_final, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
fig.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/7k_mrmr_shap_rf_d2_ss.png', dpi=400, bbox_inches='tight')
plt.close(fig)


print("\n--- Script finished successfully MRMR RF D2 SS 7 Key Features ---")