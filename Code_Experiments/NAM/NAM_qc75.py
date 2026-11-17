# -*- coding: utf-8 -*-
import os
import sys
import logging
import warnings

# --- Supressão total de ruídos e warnings ---
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("optuna").setLevel(logging.WARNING)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns
import statistics as st

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import optuna
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# ==============================================================================
# --- 1. ARQUITETURA DO NEURAL ADDITIVE MODEL (NAM) EM PYTORCH ---
# ==============================================================================
class FeatureNet(nn.Module):
    """Sub-rede MLP univariada para modelar a função de formato de cada feature individual."""
    def __init__(self, hidden_sizes, dropout):
        super().__init__()
        layers = []
        in_dim = 1
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PyTorchNAM(nn.Module):
    """Neural Additive Model: y = beta_0 + sum(f_i(x_i))"""
    def __init__(self, n_features, hidden_sizes=[64, 32], dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.feature_nets = nn.ModuleList([
            FeatureNet(hidden_sizes, dropout) for _ in range(n_features)
        ])
        self.intercept = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: [batch_size, n_features]
        out = self.intercept
        for i in range(self.n_features):
            xi = x[:, i:i+1]
            out = out + self.feature_nets[i](xi)
        return out

# ==============================================================================
# --- 2. DATA LOADING & PREPARATION ---
# ==============================================================================
print("--- Loading data NAM QC75 ---")
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
    return 1 - (numerator / denominator)

# ==============================================================================
# --- 3. OTIMIZAÇÃO COM OPTUNA ---
# ==============================================================================
GLOBAL_SEED = 23
torch.manual_seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

def objective(trial):
    # Hiperparâmetros específicos do NAM
    n_layers = trial.suggest_int('nam_n_layers', 1, 3)
    hidden_dim = trial.suggest_categorical('nam_hidden_dim', [16, 32, 64, 128])
    hidden_sizes = [hidden_dim] * n_layers
    dropout = trial.suggest_categorical('nam_dropout', [0.0, 0.05, 0.1, 0.2]) 
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])

    fold_mse, fold_rmse, fold_r2, fold_mef = [], [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for season_name, train_data, val_data in seasonal_scenarios:
        if train_data.empty or val_data.empty: 
            continue
        
        num_cols = [c for c in train_data.columns if c not in ['TIMESTAMP', 'NEE']]
        
        X_train_num = train_data[num_cols].values.astype(np.float32)
        y_train_f = train_data['NEE'].values.astype(np.float32)
        X_val_num = val_data[num_cols].values.astype(np.float32)
        y_val_f = val_data['NEE'].values.astype(np.float32)
    
        n_num_features = X_train_num.shape[1]
    
        mean_X, std_X = X_train_num.mean(axis=0), X_train_num.std(axis=0)
        std_X = np.where(std_X == 0, 1.0, std_X)
        mean_Y, std_Y = y_train_f.mean(), y_train_f.std()
        if std_Y == 0: std_Y = 1.0

        X_train_num_norm = (X_train_num - mean_X) / std_X
        Y_train_norm = (y_train_f - mean_Y) / std_Y
        X_val_num_norm = (X_val_num - mean_X) / std_X
        Y_val_norm = (y_val_f - mean_Y) / std_Y

        x_train_t = torch.as_tensor(X_train_num_norm, device=device)
        y_train_t = torch.as_tensor(Y_train_norm, device=device)
        x_val_t = torch.as_tensor(X_val_num_norm, device=device)
        y_val_t = torch.as_tensor(Y_val_norm, device=device)

        model = PyTorchNAM(
            n_features=n_num_features,
            hidden_sizes=hidden_sizes,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience, patience_counter = 40, 0
        best_weights = None
        train_indices = np.arange(len(X_train_num))
        
        for epoch in range(300):
            model.train()
            for start_idx in range(0, len(train_indices), batch_size):
                batch_idx = train_indices[start_idx:start_idx + batch_size]
                bx = x_train_t[batch_idx]
                by = y_train_t[batch_idx]

                optimizer.zero_grad()
                preds = model(bx).squeeze(-1)
                loss = F.mse_loss(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.inference_mode():
                val_preds = model(x_val_t).squeeze(-1)
                val_loss = F.mse_loss(val_preds, y_val_t).item()

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
        with torch.inference_mode():
            y_pred_norm = model(x_val_t).squeeze(-1)
        
        Y_pred_real = (y_pred_norm.cpu().numpy() * std_Y + mean_Y)
        
        mse = mean_squared_error(y_val_f, Y_pred_real)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_f, Y_pred_real)
        mef = MEF(y_val_f, Y_pred_real)
        
        fold_mse.append(mse)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        fold_mef.append(mef)
    
    if not fold_mse:
        return float('inf')

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
    
    trial.set_user_attr("list_mse", fold_mse)
    trial.set_user_attr("list_rmse", fold_rmse)
    trial.set_user_attr("list_r2", fold_r2)
    trial.set_user_attr("list_mef", fold_mef)
    
    return avg_mse
        
print("\n--- Iniciando Optuna (NAM Search) ---")

optuna.logging.set_verbosity(optuna.logging.WARNING) 
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=98))
study.optimize(objective, n_trials=200, show_progress_bar=False)

print("\nBest Trial Params: ")
best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

print("\nBest Trial Metrics (Mean ± SD across Folds):")
print(f"MSE:  {best_trial.value:.4f} ± {best_trial.user_attrs['std_mse']:.4f}")
print(f"RMSE: {best_trial.user_attrs['mean_rmse']:.4f} ± {best_trial.user_attrs['std_rmse']:.4f}")
print(f"R2:   {best_trial.user_attrs['mean_r2']:.4f} ± {best_trial.user_attrs['std_r2']:.4f}")
print(f"MEF:  {best_trial.user_attrs['mean_mef']:.4f} ± {best_trial.user_attrs['std_mef']:.4f}")

best_params = study.best_trial.params

# ==============================================================================
# --- 4. FINAL TRAINING VS TEST SET (2010-2011) ---
# ==============================================================================
Final_Train_df = pd.concat([Fold_1, Fold_2, Fold_3, Fold_4, Fold_5, Fold_6, Fold_7, Fold_8, Fold_9, Fold_10])
num_cols = [c for c in Final_Train_df.columns if c not in ['TIMESTAMP', 'NEE']]

X_train_num = Final_Train_df[num_cols].values.astype(np.float32)
y_train_final = Final_Train_df['NEE'].values.astype(np.float32)

Test_df = Data[Data['TIMESTAMP'] >= '2010-01-01'].copy() 
X_test_num = Test_df[num_cols].values.astype(np.float32)
y_test_final = Test_df['NEE'].values.astype(np.float32)

n_num_features = len(num_cols)

mean_X_final = X_train_num.mean(axis=0)
std_X_final = X_train_num.std(axis=0)
std_X_final = np.where(std_X_final == 0, 1.0, std_X_final)

mean_Y_final = y_train_final.mean()
std_Y_final = y_train_final.std()
if std_Y_final == 0: std_Y_final = 1.0

X_train_num_norm = (X_train_num - mean_X_final) / std_X_final
Y_train_final_norm = (y_train_final - mean_Y_final) / std_Y_final
X_test_num_norm = (X_test_num - mean_X_final) / std_X_final

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_x_cont = torch.as_tensor(X_train_num_norm, device=device)
train_y = torch.as_tensor(Y_train_final_norm, device=device)
test_x_cont = torch.as_tensor(X_test_num_norm, device=device)

n_layers = best_params['nam_n_layers']
hidden_dim = best_params['nam_hidden_dim']
hidden_sizes = [hidden_dim] * n_layers

model_final = PyTorchNAM(
    n_features=n_num_features,
    hidden_sizes=hidden_sizes,
    dropout=best_params['nam_dropout']
).to(device)

optimizer_final = torch.optim.AdamW(
    model_final.parameters(), 
    lr=best_params['learning_rate'], 
    weight_decay=best_params['weight_decay']
)

print("\n--- Treinando o Modelo Final (NAM em PyTorch) ---")

best_loss = float('inf')
patience = 40
min_delta = 0.001
best_weights = None
train_indices = np.arange(len(train_x_cont))
batch_size = best_params['batch_size']

for epoch in range(400):
    model_final.train()
    epoch_loss = 0.0
    n_batches = 0
    
    for start_idx in range(0, len(train_indices), batch_size):
        batch_idx = train_indices[start_idx:start_idx + batch_size]
        bx = train_x_cont[batch_idx]
        by = train_y[batch_idx]

        optimizer_final.zero_grad()
        preds = model_final(bx).squeeze(-1)
        loss = F.mse_loss(preds, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_final.parameters(), 1.0)
        optimizer_final.step()
        
        epoch_loss += loss.item()
        n_batches += 1

    avg_epoch_loss = epoch_loss / n_batches

    if avg_epoch_loss < (best_loss - min_delta):
        best_loss = avg_epoch_loss
        best_weights = {k: v.cpu().clone() for k, v in model_final.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping acionado na época {epoch+1}. Melhor loss: {best_loss:.5f}")
            break

if best_weights:
    model_final.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

model_final.eval()
with torch.inference_mode():
    y_pred_normalized = model_final(test_x_cont).squeeze(-1)

Y_test_pred_NAM = (y_pred_normalized.cpu().numpy() * std_Y_final + mean_Y_final)

df_test_final = pd.DataFrame({
    'Data': Test_df['TIMESTAMP'],
    'NEE_True': y_test_final,
    'NEE_NAM': Y_test_pred_NAM
})
df_test_final.to_csv('nam_qc75.csv', index=False)

print(f"R²:    {r2_score(y_test_final, Y_test_pred_NAM):.4f}")
print(f"MEF:   {MEF(y_test_final, Y_test_pred_NAM):.4f}")
print(f"MSE:   {mean_squared_error(y_test_final, Y_test_pred_NAM):.4f}")
print(f"RMSE:  {root_mean_squared_error(y_test_final, Y_test_pred_NAM):.4f}")

# ==============================================================================
# --- 5. GRÁFICOS DETALHADOS E INTERPRETATIVIDADE (Shape Functions do NAM) ---
# ==============================================================================
print("\n--- Gerando Gráficos de Desempenho e Curvas Aditivas do NAM ---")

Y_true = Test_df['NEE'].values
Y_predicted = Y_test_pred_NAM
dates_plot = pd.to_datetime(Test_df['TIMESTAMP']).dt.to_pydatetime()

slope, intercept = np.polyfit(Y_true, Y_predicted, 1)
equation_text = f'y = {np.round(slope,3)}x + {np.round(intercept,3)}'

fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.scatter(Y_true, Y_predicted, color='darkgreen', edgecolors=(0.1, 0.4, 0.1), alpha=0.7)
ax1.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], "k--", lw=1.5)
ax1.plot(Y_true, slope*Y_true + intercept, color='darkgreen', lw=1.5)
ax1.set_xlabel("Observed values", fontsize=12)
ax1.set_ylabel("Estimated values using NAM", fontsize=12)
ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax1.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.4)
plt.tight_layout()
fig1.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/nam_qc75_dispersion.png', dpi=400)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(10,6))
ax2.plot(dates_plot, Y_true, color='black', marker='o', markersize=6, linestyle='-', linewidth=1, label='Observed values')
ax2.plot(dates_plot, Y_predicted, color='forestgreen', marker='^', markersize=6, linestyle=':', linewidth=1.5, label='Estimated values (NAM)')
ax2.set_ylabel('NEE_VUT_REF (gC m$^{-2}$ d$^{-1}$)', fontsize=12)
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)
ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)
ax2.legend(loc='upper center', fontsize=10, bbox_to_anchor=(0.5, 1.15), ncol=2)
ax2.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.4)
plt.tight_layout()
fig2.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/nam_qc75.png', dpi=400)
plt.close(fig2)

# ==============================================================================
# --- 4. CONTINGENCY TABLE & CLASSIFICATION METRICS ---
# ==============================================================================
print("\n--- Scenario Classification Metrics (NAM) ---")

sink_plot_data = []
source_plot_data = []
Y_obs = Test_df['NEE'].values
Y_est = Y_test_pred_NAM 
model_label = 'NAM'
    
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
    save_path='/scratch/cptec/lucas.bauer/nee_ga_project/data/nam_performance_diagram_sink_qc75.png',
    cbar_label='Critical success index for sink scenarios - CSI'
)
print("Performance Diagram (Sinks) saved!")

plot_performance_diagram(
    plot_data=source_plot_data, 
    title='',
    save_path='/scratch/cptec/lucas.bauer/nee_ga_project/data/nam_performance_diagram_source_qc75.png',
    cbar_label='Critical success index for source scenarios - CSI'
)
print("Performance Diagram (Sources) saved!")

# Plotando as Funções de Resposta Univariadas (Substitui o SHAP com vantagens teóricas diretas)
fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(18, 12))
axes = axes.flatten()
model_final.eval()
with torch.inference_mode():
    for i, col_name in enumerate(num_cols):
        if i >= len(axes): break
        ax = axes[i]
        x_vals = np.linspace(X_train_num_norm[:, i].min(), X_train_num_norm[:, i].max(), 100)
        dummy_tensor = torch.zeros((100, n_num_features), device=device)
        dummy_tensor[:, i] = torch.as_tensor(x_vals, dtype=torch.float32, device=device)
        
        # Resposta da rede univariada para a feature i
        response = model_final.feature_nets[i](dummy_tensor[:, i:i+1]).cpu().numpy().squeeze()
        
        ax.plot(x_vals * std_X_final[i] + mean_X_final[i], response, color='forestgreen', lw=2)
        ax.set_title(col_name, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
fig.savefig('/scratch/cptec/lucas.bauer/nee_ga_project/data/nam_qc75_shape_functions.png', dpi=400)
plt.close(fig)

print("Experiment NAM QC75 Finished Successfully")