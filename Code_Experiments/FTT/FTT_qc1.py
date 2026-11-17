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

import torch
import torch.nn.functional as F
import torch.optim
from rtdl_revisiting_models import FTTransformer
import optuna
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from rtdl_revisiting_models import FTTransformer

import shap
import sys
# ==============================================================================
# --- 1. DATA LOADING & PREPARATION ---
# ==============================================================================
print("--- Loading data FTT QC1 Experiment---")
Data = pd.read_csv("/scratch/cptec/lucas.bauer/nee_ga_project/data/NEE_QC_1.csv", header=0, decimal='.', sep=",", na_values=-9999)
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

# ==============================================================================
# --- 3. LOOP DE OTIMIZAÇÃO COM OPTUNA E PURGED CV ---
# ==============================================================================
GLOBAL_SEED = 23
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

def objective(trial):
    # Hiperparâmetros otimizados pelo Optuna
    d_token = trial.suggest_categorical('d_token', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 4)
    n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
    d_ff = trial.suggest_categorical('d_ff', [64, 128, 256])
    dropout = trial.suggest_categorical('dropout', [0.05, 0.1, 0.2])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
    ffn_d_hidden_multiplier = trial.suggest_categorical('ffn_d_hidden_multiplier', [1.0, 2.0, 4/3])

    fold_mse, fold_rmse, fold_r2, fold_mef = [], [], [], []
    
    # Seleção de dispositivo (GPU H100 no Santos Dumont)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. AVALIA A MESMA CONFIGURAÇÃO NOS CENÁRIOS SAZONAIS
    for season_name, train_data, val_data in seasonal_scenarios:
        if train_data.empty or val_data.empty: continue
        
        cat_cols = [] 
        num_cols = [c for c in train_data.columns if c not in ['TIMESTAMP', 'NEE'] + cat_cols]
        
        X_train_num = train_data[num_cols].values.astype(np.float32)
        X_train_cat = train_data[cat_cols].values.astype(np.int64) if cat_cols else None
        y_train_f = train_data['NEE'].values.astype(np.float32)
        
        X_val_num = val_data[num_cols].values.astype(np.float32)
        X_val_cat = val_data[cat_cols].values.astype(np.int64) if cat_cols else None
        y_val_f = val_data['NEE'].values.astype(np.float32)
    
        n_cont_features = X_train_num.shape[1]
        cat_cardinalities = [] 
    
        # NORMALIZAÇÃO RIGOROSA APENAS NO TREINO DA ESTAÇÃO ATUAL
        mean_X, std_X = X_train_num.mean(axis=0), X_train_num.std(axis=0)
        std_X = np.where(std_X == 0, 1.0, std_X)
        mean_Y, std_Y = y_train_f.mean(), y_train_f.std()
        if std_Y == 0: std_Y = 1.0

        X_train_num_norm = (X_train_num - mean_X) / std_X
        Y_train_norm = (y_train_f - mean_Y) / std_Y
    
        X_val_num_norm = (X_val_num - mean_X) / std_X
        Y_val_norm = (y_val_f - mean_Y) / std_Y

        # Preparando dicionários de dados PyTorch
        train_dict = {"x_cont": torch.as_tensor(X_train_num_norm, device=device), "y": torch.as_tensor(Y_train_norm, device=device)}
        val_dict = {"x_cont": torch.as_tensor(X_val_num_norm, device=device), "y": torch.as_tensor(Y_val_norm, device=device)}
        
        if cat_cols:
            train_dict["x_cat"] = torch.as_tensor(X_train_cat, device=device)
            val_dict["x_cat"] = torch.as_tensor(X_val_cat, device=device)

        # Instanciação do FTTransformer oficial via PyTorch
        model = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=1,
            d_block=d_token,
            n_blocks=n_layers,
            attention_n_heads=n_heads,
            ffn_d_hidden_multiplier=ffn_d_hidden_multiplier,
            attention_dropout=dropout,
            ffn_dropout=dropout,
            residual_dropout=dropout
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Loop de treinamento customizado com PyTorch
        best_val_loss = float('inf')
        patience, patience_counter = 40, 0
        best_weights = None

        train_indices = np.arange(len(X_train_num))
        
        for epoch in range(400):
            model.train()
            for start_idx in range(0, len(train_indices), batch_size):
                batch_idx = train_indices[start_idx:start_idx + batch_size]
                
                batch_x_cont = train_dict["x_cont"][batch_idx]
                batch_y = train_dict["y"][batch_idx]
                batch_x_cat = train_dict.get("x_cat")[batch_idx] if cat_cols else None

                optimizer.zero_grad()
                pred = model(batch_x_cont, batch_x_cat).squeeze(-1)
                loss = F.mse_loss(pred, batch_y)
                loss.backward()
                optimizer.step()

            # Validação
            model.eval()
            with torch.no_grad():
                val_pred = model(val_dict["x_cont"], val_dict.get("x_cat")).squeeze(-1)
                val_loss = F.mse_loss(val_pred, val_dict["y"]).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_weights:
            model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

        model.eval()
        with torch.no_grad():
            y_pred_norm = model(val_dict["x_cont"], val_dict.get("x_cat")).squeeze(-1)
        
        Y_pred_real = (y_pred_norm * std_Y + mean_Y).cpu().numpy()

        fold_mse.append(mean_squared_error(y_val_f, Y_pred_real))
        fold_rmse.append(np.sqrt(mean_squared_error(y_val_f, Y_pred_real)))
        fold_r2.append(r2_score(y_val_f, Y_pred_real))
        fold_mef.append(MEF(y_val_f, Y_pred_real))
    
    avg_mse = np.mean(fold_mse) if fold_mse else float('inf')
    
    # --- PRINT DO RESUMO DO TRIAL ---
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
    
    trial.set_user_attr("list_mse", fold_mse)
    trial.set_user_attr("list_rmse", fold_rmse)
    trial.set_user_attr("list_r2", fold_r2)
    trial.set_user_attr("list_mef", fold_mef)
    
    return avg_mse

print("\n--- Iniciando Optuna Optimization ---")
optuna.logging.set_verbosity(optuna.logging.WARNING) 
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=98))
study.optimize(objective, n_trials=200)

trials_df_ftt_d1_cv = study.trials_dataframe()
#trials_df_tft_d1_cv.to_csv('/scratch/cptec/lucas.bauer/nee_ga_project/data/trials_df_FTT_d1.csv', index=False, sep=',')

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

best_params = study.best_trial.params
# ==============================================================================
# --- 3. FINAL TRAINING VS TEST SET (2010-2011) ---
# ==============================================================================

# 1. Preparação dos DataFrames de Treino Final e Teste
Final_Train_df = pd.concat([Fold_1, Fold_2, Fold_3, Fold_4, Fold_5, Fold_6, Fold_7, Fold_8, Fold_9, Fold_10])

cat_cols = [] # Ex: ['CAT_FEATURE_1'] se houver, senão lista vazia
num_cols = [c for c in Final_Train_df.columns if c not in ['TIMESTAMP', 'NEE'] + cat_cols]

X_train_num = Final_Train_df[num_cols].values.astype(np.float32)
X_train_cat = Final_Train_df[cat_cols].values.astype(np.int64) if cat_cols else None
y_train_final = Final_Train_df['NEE'].values.astype(np.float32)

Test_df = Data[Data['TIMESTAMP'] >= '2010-01-01'].copy() 
X_test_num = Test_df[num_cols].values.astype(np.float32)
X_test_cat = Test_df[cat_cols].values.astype(np.int64) if cat_cols else None
y_test_final = Test_df['NEE'].values.astype(np.float32)

n_cont_features = len(num_cols)
cat_cardinalities = []

# 2. Normalização rigorosa baseada exclusivamente no treino final (Z-score)
mean_X_final = X_train_num.mean(axis=0)
std_X_final = X_train_num.std(axis=0)
std_X_final = np.where(std_X_final == 0, 1.0, std_X_final)

mean_Y_final = y_train_final.mean()
std_Y_final = y_train_final.std()
if std_Y_final == 0: std_Y_final = 1.0

X_train_num_norm = (X_train_num - mean_X_final) / std_X_final
Y_train_final_norm = (y_train_final - mean_Y_final) / std_Y_final

X_test_num_norm = (X_test_num - mean_X_final) / std_X_final
# O y_test não é normalizado na entrada, pois compararemos na escala real

# 3. Alocação direta para a GPU (H100) via PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_x_cont = torch.as_tensor(X_train_num_norm, device=device)
train_y = torch.as_tensor(Y_train_final_norm, device=device)
train_x_cat = torch.as_tensor(X_train_cat, device=device) if cat_cols else None

test_x_cont = torch.as_tensor(X_test_num_norm, device=device)
test_y = torch.as_tensor(y_test_final, device=device) # Escala real para teste
test_x_cat = torch.as_tensor(X_test_cat, device=device) if cat_cols else None

# 4. Instanciação e Otimizador do Modelo Final em PyTorch
model_final = FTTransformer(
    n_cont_features=n_cont_features,
    cat_cardinalities=cat_cardinalities,
    d_out=1,
    d_block=best_params['d_token'],    
    n_blocks=best_params['n_layers'],  
    attention_n_heads=best_params['n_heads'],
    ffn_d_hidden_multiplier=best_params['ffn_d_hidden_multiplier'],  
    attention_dropout=best_params['dropout'],
    ffn_dropout=best_params['dropout'],
    residual_dropout=best_params['dropout']
).to(device)

optimizer_final = torch.optim.AdamW(
    model_final.parameters(), 
    lr=best_params['learning_rate'], 
    weight_decay=best_params['weight_decay']
)

print("\n--- Treinando o Modelo Final (FT-Transformer em PyTorch) ---")

best_loss = float('inf')
patience = 50
patience_counter = 0
min_delta = 0.001
best_weights = None

train_indices = np.arange(len(train_x_cont))
batch_size = best_params['batch_size']

# Loop de treinamento final
for epoch in range(500):
    model_final.train()
    epoch_loss = 0.0
    n_batches = 0
    
    # Mini-batch loop (mantendo shuffle=False para preservar autocorrelação temporal)
    for start_idx in range(0, len(train_indices), batch_size):
        batch_idx = train_indices[start_idx:start_idx + batch_size]
        
        batch_x_cont = train_x_cont[batch_idx]
        batch_y = train_y[batch_idx]
        batch_x_cat = train_x_cat[batch_idx] if train_x_cat is not None else None

        optimizer_final.zero_grad()
        pred = model_final(batch_x_cont, batch_x_cat).squeeze(-1)
        loss = F.mse_loss(pred, batch_y)
        loss.backward()
        optimizer_final.step()
        
        epoch_loss += loss.item()
        n_batches += 1

    avg_epoch_loss = epoch_loss / n_batches

    # Early Stopping manual monitorando a loss de treino
    if avg_epoch_loss < (best_loss - min_delta):
        best_loss = avg_epoch_loss
        best_weights = {k: v.cpu().clone() for k, v in model_final.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping acionado na época {epoch+1}. Melhor loss: {best_loss:.5f}")
            break

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/500 - Loss: {avg_epoch_loss:.5f}")

# Carrega os melhores pesos do treinamento final
if best_weights:
    model_final.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

# 5. Predição no conjunto de teste (2010+) direto na GPU
model_final.eval()
with torch.no_grad():
    y_pred_normalized = model_final(test_x_cont, test_x_cat).squeeze(-1)

# Reversão da normalização para a escala real do NEE (tudo na GPU, convertendo para numpy só no final para plot/métricas)
Y_test_pred_FT = (y_pred_normalized * std_Y_final + mean_Y_final).cpu().numpy()

df_test_final = pd.DataFrame({
    'Data': Test_df['TIMESTAMP'],
    'NEE_True': y_test_final,
    'NEE_FTT': Y_test_pred_FT
})

df_test_final.to_csv('FTT_qc1.csv', index=False)

print(f"R²:    {r2_score(y_test_final, Y_test_pred_FT):.4f}")
print(f"MEF:   {MEF(y_test_final, Y_test_pred_FT):.4f}")
print(f"MSE:   {mean_squared_error(y_test_final, Y_test_pred_FT):.4f}")
print(f"RMSE:  {root_mean_squared_error(y_test_final, Y_test_pred_FT):.4f}")

# ==============================================================================
# --- 3.1. DETAILED FTT PERFORMANCE PLOTS (Scatter & Time Series) ---
# ==============================================================================
print("\n--- Gerando Gráficos Detalhados do FTT ---")

Y_true = Test_df['NEE'].values
Y_predicted = Y_test_pred_FT
dates_plot = pd.to_datetime(Test_df['TIMESTAMP']).dt.to_pydatetime()

# 1. Gráfico de Dispersão (Regressão Linear)
slope, intercept = np.polyfit(Y_true, Y_predicted, 1)
equation_text = f'y = {np.round(slope,3)}x + {np.round(intercept,3)}'

fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.scatter(Y_true, Y_predicted, color='purple', edgecolors=(0.82, 0.12, 0.12), alpha=0.7)
ax1.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], "k--", lw=1.5)
ax1.plot(Y_true, slope*Y_true + intercept, color='purple', lw=1.5)

ax1.set_xlabel("Observed values", fontsize=12)
ax1.set_ylabel("Estimated values using FTT", fontsize=12)
ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax1.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.4)
ax1.legend(loc='lower right')
plt.tight_layout()
fig1.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/ftt_qc1_dispersion.png', dpi=400)
plt.close(fig1)

# 2. Gráfico de Série Temporal Específico (FTT vs Observado)
fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.plot(dates_plot, Y_true, color='black', marker='o', markersize=6, linestyle='-', linewidth=1, label='Observed values')
ax2.plot(dates_plot, Y_predicted, color='red', marker='^', markersize=6, linestyle=':', linewidth=1.5, label='Estimated values')
ax2.set_ylabel('NEE_VUT_REF (gC m$^{-2}$ d$^{-1}$)', fontsize=12)

locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)

ax2.legend(loc='upper center', fontsize=10, bbox_to_anchor=(0.5, 1.15), ncol=2)
ax2.tick_params(axis='both', which='major', labelsize=10)
ax2.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.4)
plt.tight_layout()
fig2.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/ftt_qc1.png', dpi=400)
plt.close(fig2)

# ==============================================================================
# --- 4. CONTINGENCY TABLE & CLASSIFICATION METRICS ---
# ==============================================================================
print("\n--- Scenario Classification Metrics (FTT) ---")

sink_plot_data = []
source_plot_data = []
Y_obs = Test_df['NEE'].values
Y_est = Y_test_pred_FT 
model_label = 'FTT'
    
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
        ax.scatter(data['SR'], data['POD'], c='purple', marker='*', edgecolor='black', label=lbl, s=400, zorder=101)

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
    save_path='/scratch/cptec/lucas.bauer/nee_ga_project/data/pftt_performance_diagram_sink_qc1.png',
    cbar_label='Critical success index for sink scenarios - CSI'
)
print("Performance Diagram (Sinks) saved!")

plot_performance_diagram(
    plot_data=source_plot_data, 
    title='',
    save_path='/scratch/cptec/lucas.bauer/nee_ga_project/data/pftt_performance_diagram_source_qc1.png',
    cbar_label='Critical success index for source scenarios - CSI'
)
print("Performance Diagram (Sources) saved!")

# ==============================================================================
# --- 6. Análise SHAP ---
# ==============================================================================
print("\n--- Iniciando Análise SHAP no Conjunto de Treinamento ---")

# Adaptação para as variáveis do seu script PyTorch
feature_names = num_cols
X_train_np = X_train_num_norm  # Já é o array numpy normalizado

# Função wrapper para o KernelExplainer interagir com o modelo PyTorch na GPU
def predict_wrapper(X_np):
    model_final.eval()
    with torch.no_grad():
        x_cont_tensor = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        x_cat_tensor = torch.as_tensor(X_train_cat, device=device) if X_train_cat is not None else None
        
        preds = model_final(x_cont_tensor, x_cat_tensor).squeeze(-1)
        return preds.cpu().numpy()

# Omitindo warnings do Matplotlib que poluem o log
mpl.rc('figure', max_open_warning=0)

# Redirecionando TUDO para o "buraco negro" antes do SHAP rodar
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

try:
    # Usando o wrapper customizado para o modelo PyTorch
    explainer_train = shap.KernelExplainer(predict_wrapper, X_train_np) 
    shap_values_train = explainer_train.shap_values(X_train_np, silent=True)
    
    if isinstance(shap_values_train, list):
        shap_values_train = shap_values_train[0]
        
    if len(shap_values_train.shape) == 3:
        shap_values_train = shap_values_train[:, :, 0]

finally:
    # Devolvendo a voz ao Python após o SHAP terminar, não importa o que aconteça
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr

# --- SHAP Plots ---
print("Salvando gráficos SHAP no diretório...")

fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_train, X_train_np, feature_names=feature_names, show=False, max_display=10)

plt.xlabel(r"SHAP values (Contributions to NEE, gC $m^{-2}$ $d^{-1}$)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("", fontsize=14)
plt.tight_layout()

fig.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/beeswarm_ftt_qc1.png', dpi=400, bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_train, X_train_np, feature_names=feature_names, plot_type="bar", show=False, max_display=10)
plt.xlabel(r"Mean (|SHAP value|) (Mean contributions to NEE, gC $m^{-2}$ $d^{-1}$)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("", fontsize=14)
plt.tight_layout()

fig.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/shap_bar_ftt_qc1.png', dpi=400, bbox_inches='tight')
plt.close(fig)

print("Experiment FTT QC1 Finished")