import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import time
from torch.nn.parameter import Parameter
import torch.optim as optim
import random
import pandas as pd
torch.set_default_dtype(torch.float32)
from scipy import sparse
from sklearn.linear_model import Ridge
import causaltensor as ct
from sklearn.decomposition import KernelPCA

# -----------------------------
# Simulation Settings
# -----------------------------
np.random.seed(4)
N, T, r, P = 100, 200, 4, 3       # N: units, T: time periods, r: latent dim, P: covariates
lambda_ntk = 1                 # regularization for NTK ridge regression (if needed)
epochs = 200
lr_model = 0.0005
num_simulations = 50
noise_std = 0.5
def compute_se(metric_list):
    """
    metric_list: list of tuples (est_ATE, MAE_comp, MAE_treat, MSE_comp, MSE_treat)
    returns: array of SEs for each metric
    """
    arr = np.array(metric_list)  # shape (S,5)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return se


# -----------------------------
# Data Generating Functions for h(F)
# -----------------------------
def h_linear(Lambda_i, F_t, C):
    return C * np.dot(Lambda_i, F_t)

def h_sine(Lambda_i, F_t, C):
    return C * np.sin(np.dot(Lambda_i, F_t))

def h_polynomial(Lambda_i, F_t, C1, C2):
    dot_val = np.dot(Lambda_i, F_t)
    return C1 * dot_val + C2 * (dot_val ** 2)

# -----------------------------
# New ReLU Generation Function
# -----------------------------
def generate_unit_output(F, X_i, hidden_dim=10):
    # F is assumed to be (r, T) so that W1 @ F yields (hidden_dim, T)
    W1, b1 = np.random.randn(hidden_dim, r) * 0.5, np.random.randn(hidden_dim, 1) * 0.5
    W2, b2 = np.random.randn(1, hidden_dim) * 0.5, np.random.randn(1, 1) * 0.5
    H = np.maximum(W1 @ F + b1, 0)      # H: (hidden_dim, T)
    h_F = (W2 @ H + b2).flatten()       # h_F: (T,)
    return h_F

# -----------------------------
# Data Generating Function that Selects a Model
# -----------------------------
def generate_unit_output_model(model_type, Lambda_i, F, X_i, noise_std=0.5):
    # F is assumed to be (T x r)
    T_local = F.shape[0]
    y = np.zeros(T_local)
    if model_type == 'linear':
        C = np.random.randn() * 0.5
        for t in range(T_local):
            y[t] = h_linear(Lambda_i, F[t], C)
    elif model_type == 'sine':
        C = np.random.randn() * 2
        for t in range(T_local):
            y[t] = h_sine(Lambda_i, F[t], C)
    elif model_type == 'polynomial':
        C1 = np.random.randn() * 0.2
        C2 = np.random.randn() * 0.2
        for t in range(T_local):
            y[t] = h_polynomial(Lambda_i, F[t], C1, C2)
    elif model_type == 'relu':
        y = generate_unit_output(F.T, X_i, hidden_dim=10)
    else:
        raise ValueError("Invalid model type")
    # Add noise (independent of the covariate effect)
    y = y
    return y

# -----------------------------
# Autoencoder Models
# -----------------------------
class DisjointDecoderAE(nn.Module):
    def __init__(self, num_units, latent_dim, enc_hidden=[64,64,64], dec_hidden=[64,64,64]):
        super().__init__()
        enc_layers, input_dim = [], num_units
        for h in enc_hidden:
            enc_layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        enc_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        self.decoders = nn.ModuleList()
        for _ in range(num_units):
            dec_layers, input_dim = [], latent_dim
            for h in dec_hidden:
                dec_layers += [nn.Linear(input_dim, h), nn.ReLU()]
                input_dim = h
            dec_layers.append(nn.Linear(input_dim, 1))
            self.decoders.append(nn.Sequential(*dec_layers))
    
    def forward(self, x):
        z = self.encoder(x)
        return torch.cat([d(z) for d in self.decoders], dim=1)

class SingleDecoderAE(nn.Module):
    def __init__(self, num_units, latent_dim, enc_hidden=[16,16,16], dec_hidden=[16,16,16]):
        super().__init__()
        enc_layers, input_dim = [], num_units
        for h in enc_hidden:
            enc_layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        enc_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
    
        dec_layers, input_dim = [], latent_dim
        for h in dec_hidden:
            dec_layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        dec_layers.append(nn.Linear(input_dim, num_units))
        self.decoder = nn.Sequential(*dec_layers)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class SimpleNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(SimpleNNRegressor, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)
        # Initialize weights similarly to our generation function
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x):
        # x: (batch_size, P)
        h = self.relu(self.hidden(x))
        out = self.output(h)
        return out

def train_simple_nn_regressor(X_train, y_train, num_epochs=1500, lr=0.001, verbose=False):
    input_dim = X_train.shape[1]
    model = SimpleNNRegressor(input_dim, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 300 == 0:
            print(f"[Simple NN] Epoch {epoch} | Loss: {loss.item():.4f}")
    return model

def train_model(model, X, Y, mask, epochs=1000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = (criterion(pred, Y) * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
    return deepcopy(model.eval())

def impute_disjoint(model, X_train, treated_units, control_units, post_periods):
    model.eval()
    with torch.no_grad():
        F_hat_b = []
        for t in post_periods:
            x_input = np.zeros(N)
            x_input[control_units] = X_train[control_units, t]
            x_tensor = torch.tensor(x_input, dtype=torch.float).unsqueeze(0)
            F_hat_b.append(model.encoder(x_tensor))
        F_hat_b = torch.cat(F_hat_b, dim=0)
        Y_d_hat = np.array([model.decoders[unit](F_hat_b).squeeze(-1).numpy() for unit in treated_units])
    return Y_d_hat

def add_covariates_back(Y_residual_hat, treated_units, post_periods, X, control_units, Y_obs):
    Y_final = np.zeros_like(Y_residual_hat)
    for idx, t in enumerate(post_periods):
        model_lr = LinearRegression().fit(X[control_units], Y_obs[control_units, t])
        g_hat_treated = model_lr.predict(X[treated_units])
        Y_final[:, idx] = Y_residual_hat[:, idx] + g_hat_treated
    return Y_final

def compute_metrics(Y_true0, Y_true1, Y_hat, tau_it):
    N2, T2 = Y_true0.shape
    est_ATE = (Y_true1 - Y_hat).mean()
    est_ATE_unit = np.mean(Y_true1 - Y_hat, axis=1).reshape(-1, 1)
    MAE_comp = np.abs(Y_true0 - Y_hat).mean()
    MAE_treat = np.abs((Y_true1 - Y_hat) - tau_it).mean()
    MAE_unit = np.abs(est_ATE_unit - tau_it).mean()
    MAE_ave = np.abs(est_ATE-np.mean(tau_it))
    MSE_comp = np.linalg.norm(Y_true0 - Y_hat, 'fro')**2 / (N2 * T2)
    MSE_treat = ((tau_it - (Y_true1 - Y_hat))**2).mean()
    MSE_unit = ((tau_it - est_ATE_unit)**2).mean()
    MSE_ave = (est_ATE-np.mean(tau_it))**2
    return est_ATE, MAE_comp, MAE_treat, MSE_comp, MSE_treat, MAE_unit, MSE_unit, MAE_ave, MSE_ave

from torch.nn.modules.activation import ReLU
#############################################
############ code of AEMC-NE ################
#############################################
# The code of AEMC-NE is cited from: https://openreview.net/attachment?id=kPrxk6tUcg&name=supplementary_material
class RCAutoRec(nn.Module):
    def __init__(self, dim_in, layerwise_hidden_dim, elementwise_hidden_dim, mod):
        super(RCAutoRec, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_in
        self.mod = mod
        actfun_LW = nn.Tanh()
        actfun_EW = nn.Tanh()
        # Layer-wise network
        self.LNet = nn.Sequential()
        nl_layerwise = len(layerwise_hidden_dim)
        self.LNet.add_module("L1", nn.Linear(self.dim_in, layerwise_hidden_dim[0]))
        self.LNet.add_module("L1_actf", actfun_LW)
        for i in range(1, nl_layerwise):
            self.LNet.add_module("L"+str(i+1), nn.Linear(layerwise_hidden_dim[i-1], layerwise_hidden_dim[i]))
            self.LNet.add_module("L"+str(i+1)+"_actf", actfun_LW)
        self.LNet.add_module("L"+str(nl_layerwise+1), nn.Linear(layerwise_hidden_dim[-1], self.dim_out))
        # Element-wise network
        self.ENet = nn.Sequential()
        nl_elementwise = len(elementwise_hidden_dim)
        self.ENet.add_module("E1", nn.Linear(1, elementwise_hidden_dim[0]))
        self.ENet.add_module("E1_actf", actfun_EW)
        for i in range(1, nl_elementwise):
            self.ENet.add_module("E"+str(i+1), nn.Linear(elementwise_hidden_dim[i-1], elementwise_hidden_dim[i]))
            self.ENet.add_module("E"+str(i+1)+"_actf", actfun_EW)
        self.ENet.add_module("E"+str(nl_elementwise+1), nn.Linear(elementwise_hidden_dim[-1], 1))
        
    def forward(self, x, mask):
        x = self.LNet(x)
        if self.mod == 'NE':
            train_num = int(torch.sum(mask))
            y = x[mask > 0].reshape(train_num, 1)
            y = self.ENet(y) + y * 0.8
            x[mask > 0] = y.reshape(train_num)
        return x
    
    def evaluate(self, x, x_test, mask):
        prediction = self.forward(x, mask)
        RMSE_squa = float(torch.norm(torch.mul(x_test - prediction, mask))**2 / (torch.sum(mask > 0)))
        RMSE_squa = RMSE_squa / float(torch.norm(torch.mul(x_test, mask))**2 / (torch.sum(mask > 0)))
        return RMSE_squa ** (1/2)

def training_RCAutoRec(mr, mod, main_hidden_dim, act_hidden_dim, epoch, lr, wd, device, data_matrix, mask_matrix, test_matrix):
    data_matrix = torch.FloatTensor(data_matrix).to(device)
    mask_matrix = torch.FloatTensor(mask_matrix).to(device)
    test_matrix = torch.FloatTensor(test_matrix).to(device)
    m_sample, dim_in = data_matrix.shape 
    net = RCAutoRec(dim_in, main_hidden_dim, act_hidden_dim, mod).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    RMSE_list = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    for i in range(epoch):
        optimizer.zero_grad()
        x_pred = net.forward(data_matrix, mask_matrix)
        loss = torch.norm(torch.mul(data_matrix - x_pred, mask_matrix))**2 / mask_matrix.sum()
        RMSE = 0
        if i >= 1:
            RMSE = net.evaluate(data_matrix, test_matrix, mask_matrix)
        loss.backward()
        optimizer.step()
        scheduler.step()
        RMSE_list.append(RMSE)
    return net, RMSE_list[-1], min(RMSE_list), RMSE_list.index(min(RMSE_list))


def did_impute(Y_obs, treated_units, control_units, pre_periods, post_periods):
    pre_baseline =  np.mean(Y_obs[control_units[:, None], pre_periods], axis=1)
    trend = np.array([np.mean(Y_obs[control_units, t] - pre_baseline) for t in post_periods])
    baseline_treated = np.mean(Y_obs[treated_units[:, None], pre_periods], axis=1)
    Y_hat_did = np.outer(baseline_treated, np.ones(len(post_periods))) + trend
    return Y_hat_did


def vertical_regression_impute(Y_obs, treated_units, control_units, pre_periods, post_periods):
    Y_hat = np.zeros((len(treated_units), len(post_periods)))
    for idx, unit in enumerate(treated_units):
        X_reg = Y_obs[control_units[:,None], pre_periods]
        y_reg = Y_obs[unit, pre_periods]
        model_lr_vertical = LinearRegression().fit(X_reg.T, y_reg)
        Ythat = []
        for t in post_periods:
            X_pred = Y_obs[control_units[:,None], t]
            Y_pred = X_pred.T.dot(model_lr_vertical.coef_)+ model_lr_vertical.intercept_
            Ythat.append(Y_pred.item())
        Y_hat[idx, :] = Ythat
    return Y_hat
    
def generate_nonlinear_covariate_effect(X, T, hidden_dim=32):
    """
    Generate a covariate effect matrix (N, T) by applying a one-hidden-layer ReLU NN to X.
    This mimics the transformation used in many NN models.
    """
    N, P = X.shape
    effect = np.zeros((N, T))
    for t in range(T):
        # Use Xavier initialization
        W1 = np.random.randn(P, hidden_dim) * np.sqrt(2. / P)
        b1 = np.random.randn(1, hidden_dim)
        W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
        b2 = np.random.randn(1, 1)
        hidden = np.maximum(X.dot(W1) + b1, 0)
        effect[:, t:t+1] = hidden.dot(W2) + b2
    return effect

def generate_nonlinear_covariate_effect_sinusoidal(X, T):
    """
    Generate a covariate effect matrix (N, T) using a sinusoidal transformation.

    For each time step t, a random weight vector and bias are drawn so that:

        effect[i, t] = sin( X[i,:] dot w_t + b_t )

    Parameters:
        X: numpy array of shape (N, P)
        T: number of time periods

    Returns:
        effect: numpy array of shape (N, T)
    """
    N, P = X.shape
    effect = np.zeros((N, T))
    for t in range(T):
        w_t = np.random.randn(P)
        b_t = np.random.randn()
        effect[:, t] = np.tanh(np.abs(X.dot(w_t)))**0.5 + b_t
        #effect[:, t] = np.abs(X.dot(w_t))**0.5 + b_t
        #effect[:, t] = np.log(np.abs(X.dot(w_t) + b_t))
    return effect

# -----------------------------
# Outer Loop over Data-Generating Models
# -----------------------------
model_types = ['linear', 'sine', 'polynomial', 'relu']

all_metrics = {
    mtype: {
       'K_PCA':[]
    }
    for mtype in model_types
}

# RCAutoRec parameters
mr = 0.5
mod = 'NE'
main_hidden_dim = [30, 30]
act_hidden_dim = [20, 20]
rc_epoch = 30
rc_lr = 0.001
rc_wd = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_se(metric_list):
    """
    metric_list: list of tuples (est_ATE, MAE_comp, MAE_treat, MSE_comp, MSE_treat)
    returns: array of SEs for each metric
    """
    arr = np.array(metric_list)  # shape (S,5)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return se


# -----------------------------
# Data Generating Functions for h(F)
# -----------------------------
def h_linear(Lambda_i, F_t, C):
    return C * np.dot(Lambda_i, F_t)

def h_sine(Lambda_i, F_t, C):
    return C * np.sin(np.dot(Lambda_i, F_t))

def h_polynomial(Lambda_i, F_t, C1, C2):
    dot_val = np.dot(Lambda_i, F_t)
    return C1 * dot_val + C2 * (dot_val ** 2)

# -----------------------------
# New ReLU Generation Function
# -----------------------------
def generate_unit_output(F, X_i, hidden_dim=10):
    # F is assumed to be (r, T) so that W1 @ F yields (hidden_dim, T)
    W1, b1 = np.random.randn(hidden_dim, r) * 0.5, np.random.randn(hidden_dim, 1) * 0.5
    W2, b2 = np.random.randn(1, hidden_dim) * 0.5, np.random.randn(1, 1) * 0.5
    H = np.maximum(W1 @ F + b1, 0)      # H: (hidden_dim, T)
    h_F = (W2 @ H + b2).flatten()       # h_F: (T,)
    return h_F

# -----------------------------
# Data Generating Function that Selects a Model
# -----------------------------
def generate_unit_output_model(model_type, Lambda_i, F, X_i, noise_std=0.5):
    # F is assumed to be (T x r)
    T_local = F.shape[0]
    y = np.zeros(T_local)
    if model_type == 'linear':
        C = np.random.randn() * 0.5
        for t in range(T_local):
            y[t] = h_linear(Lambda_i, F[t], C)
    elif model_type == 'sine':
        C = np.random.randn() * 2
        for t in range(T_local):
            y[t] = h_sine(Lambda_i, F[t], C)
    elif model_type == 'polynomial':
        C1 = np.random.randn() * 0.2
        C2 = np.random.randn() * 0.2
        for t in range(T_local):
            y[t] = h_polynomial(Lambda_i, F[t], C1, C2)
    elif model_type == 'relu':
        y = generate_unit_output(F.T, X_i, hidden_dim=10)
    else:
        raise ValueError("Invalid model type")
    # Add noise (independent of the covariate effect)
    y = y
    return y

# -----------------------------
# Autoencoder Models
# -----------------------------
class DisjointDecoderAE(nn.Module):
    def __init__(self, num_units, latent_dim, enc_hidden=[64,64,64], dec_hidden=[64,64,64]):
        super().__init__()
        enc_layers, input_dim = [], num_units
        for h in enc_hidden:
            enc_layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        enc_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        self.decoders = nn.ModuleList()
        for _ in range(num_units):
            dec_layers, input_dim = [], latent_dim
            for h in dec_hidden:
                dec_layers += [nn.Linear(input_dim, h), nn.ReLU()]
                input_dim = h
            dec_layers.append(nn.Linear(input_dim, 1))
            self.decoders.append(nn.Sequential(*dec_layers))
    
    def forward(self, x):
        z = self.encoder(x)
        return torch.cat([d(z) for d in self.decoders], dim=1)

class SingleDecoderAE(nn.Module):
    def __init__(self, num_units, latent_dim, enc_hidden=[16,16,16], dec_hidden=[16,16,16]):
        super().__init__()
        enc_layers, input_dim = [], num_units
        for h in enc_hidden:
            enc_layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        enc_layers.append(nn.Linear(input_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
    
        dec_layers, input_dim = [], latent_dim
        for h in dec_hidden:
            dec_layers += [nn.Linear(input_dim, h), nn.ReLU()]
            input_dim = h
        dec_layers.append(nn.Linear(input_dim, num_units))
        self.decoder = nn.Sequential(*dec_layers)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class SimpleNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(SimpleNNRegressor, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)
        # Initialize weights similarly to our generation function
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x):
        # x: (batch_size, P)
        h = self.relu(self.hidden(x))
        out = self.output(h)
        return out

def train_simple_nn_regressor(X_train, y_train, num_epochs=1500, lr=0.001, verbose=False):
    input_dim = X_train.shape[1]
    model = SimpleNNRegressor(input_dim, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = loss_fn(preds, y_tensor)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 300 == 0:
            print(f"[Simple NN] Epoch {epoch} | Loss: {loss.item():.4f}")
    return model

def train_model(model, X, Y, mask, epochs=1000, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = (criterion(pred, Y) * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
    return deepcopy(model.eval())

def impute_disjoint(model, X_train, treated_units, control_units, post_periods):
    model.eval()
    with torch.no_grad():
        F_hat_b = []
        for t in post_periods:
            x_input = np.zeros(N)
            x_input[control_units] = X_train[control_units, t]
            x_tensor = torch.tensor(x_input, dtype=torch.float).unsqueeze(0)
            F_hat_b.append(model.encoder(x_tensor))
        F_hat_b = torch.cat(F_hat_b, dim=0)
        Y_d_hat = np.array([model.decoders[unit](F_hat_b).squeeze(-1).numpy() for unit in treated_units])
    return Y_d_hat

def add_covariates_back(Y_residual_hat, treated_units, post_periods, X, control_units, Y_obs):
    Y_final = np.zeros_like(Y_residual_hat)
    for idx, t in enumerate(post_periods):
        model_lr = LinearRegression().fit(X[control_units], Y_obs[control_units, t])
        g_hat_treated = model_lr.predict(X[treated_units])
        Y_final[:, idx] = Y_residual_hat[:, idx] + g_hat_treated
    return Y_final

def compute_metrics(Y_true0, Y_true1, Y_hat, tau_it):
    N2, T2 = Y_true0.shape
    est_ATE = (Y_true1 - Y_hat).mean()
    est_ATE_unit = np.mean(Y_true1 - Y_hat, axis=1).reshape(-1, 1)
    MAE_comp = np.abs(Y_true0 - Y_hat).mean()
    MAE_treat = np.abs((Y_true1 - Y_hat) - tau_it).mean()
    MAE_unit = np.abs(est_ATE_unit - tau_it).mean()
    MAE_ave = np.abs(est_ATE-np.mean(tau_it))
    MSE_comp = np.linalg.norm(Y_true0 - Y_hat, 'fro')**2 / (N2 * T2)
    MSE_treat = ((tau_it - (Y_true1 - Y_hat))**2).mean()
    MSE_unit = ((tau_it - est_ATE_unit)**2).mean()
    MSE_ave = (est_ATE-np.mean(tau_it))**2
    return est_ATE, MAE_comp, MAE_treat, MSE_comp, MSE_treat, MAE_unit, MSE_unit, MAE_ave, MSE_ave

from torch.nn.modules.activation import ReLU
#############################################
############ code of AEMC-NE ################
#############################################
# The code of AEMC-NE is cited from: https://openreview.net/attachment?id=kPrxk6tUcg&name=supplementary_material
class RCAutoRec(nn.Module):
    def __init__(self, dim_in, layerwise_hidden_dim, elementwise_hidden_dim, mod):
        super(RCAutoRec, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_in
        self.mod = mod
        actfun_LW = nn.Tanh()
        actfun_EW = nn.Tanh()
        # Layer-wise network
        self.LNet = nn.Sequential()
        nl_layerwise = len(layerwise_hidden_dim)
        self.LNet.add_module("L1", nn.Linear(self.dim_in, layerwise_hidden_dim[0]))
        self.LNet.add_module("L1_actf", actfun_LW)
        for i in range(1, nl_layerwise):
            self.LNet.add_module("L"+str(i+1), nn.Linear(layerwise_hidden_dim[i-1], layerwise_hidden_dim[i]))
            self.LNet.add_module("L"+str(i+1)+"_actf", actfun_LW)
        self.LNet.add_module("L"+str(nl_layerwise+1), nn.Linear(layerwise_hidden_dim[-1], self.dim_out))
        # Element-wise network
        self.ENet = nn.Sequential()
        nl_elementwise = len(elementwise_hidden_dim)
        self.ENet.add_module("E1", nn.Linear(1, elementwise_hidden_dim[0]))
        self.ENet.add_module("E1_actf", actfun_EW)
        for i in range(1, nl_elementwise):
            self.ENet.add_module("E"+str(i+1), nn.Linear(elementwise_hidden_dim[i-1], elementwise_hidden_dim[i]))
            self.ENet.add_module("E"+str(i+1)+"_actf", actfun_EW)
        self.ENet.add_module("E"+str(nl_elementwise+1), nn.Linear(elementwise_hidden_dim[-1], 1))
        
    def forward(self, x, mask):
        x = self.LNet(x)
        if self.mod == 'NE':
            train_num = int(torch.sum(mask))
            y = x[mask > 0].reshape(train_num, 1)
            y = self.ENet(y) + y * 0.8
            x[mask > 0] = y.reshape(train_num)
        return x
    
    def evaluate(self, x, x_test, mask):
        prediction = self.forward(x, mask)
        RMSE_squa = float(torch.norm(torch.mul(x_test - prediction, mask))**2 / (torch.sum(mask > 0)))
        RMSE_squa = RMSE_squa / float(torch.norm(torch.mul(x_test, mask))**2 / (torch.sum(mask > 0)))
        return RMSE_squa ** (1/2)

def training_RCAutoRec(mr, mod, main_hidden_dim, act_hidden_dim, epoch, lr, wd, device, data_matrix, mask_matrix, test_matrix):
    data_matrix = torch.FloatTensor(data_matrix).to(device)
    mask_matrix = torch.FloatTensor(mask_matrix).to(device)
    test_matrix = torch.FloatTensor(test_matrix).to(device)
    m_sample, dim_in = data_matrix.shape 
    net = RCAutoRec(dim_in, main_hidden_dim, act_hidden_dim, mod).to(device)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    RMSE_list = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    for i in range(epoch):
        optimizer.zero_grad()
        x_pred = net.forward(data_matrix, mask_matrix)
        loss = torch.norm(torch.mul(data_matrix - x_pred, mask_matrix))**2 / mask_matrix.sum()
        RMSE = 0
        if i >= 1:
            RMSE = net.evaluate(data_matrix, test_matrix, mask_matrix)
        loss.backward()
        optimizer.step()
        scheduler.step()
        RMSE_list.append(RMSE)
    return net, RMSE_list[-1], min(RMSE_list), RMSE_list.index(min(RMSE_list))


def did_impute(Y_obs, treated_units, control_units, pre_periods, post_periods):
    pre_baseline =  np.mean(Y_obs[control_units[:, None], pre_periods], axis=1)
    trend = np.array([np.mean(Y_obs[control_units, t] - pre_baseline) for t in post_periods])
    baseline_treated = np.mean(Y_obs[treated_units[:, None], pre_periods], axis=1)
    Y_hat_did = np.outer(baseline_treated, np.ones(len(post_periods))) + trend
    return Y_hat_did


def vertical_regression_impute(Y_obs, treated_units, control_units, pre_periods, post_periods):
    Y_hat = np.zeros((len(treated_units), len(post_periods)))
    for idx, unit in enumerate(treated_units):
        X_reg = Y_obs[control_units[:,None], pre_periods]
        y_reg = Y_obs[unit, pre_periods]
        model_lr_vertical = LinearRegression().fit(X_reg.T, y_reg)
        Ythat = []
        for t in post_periods:
            X_pred = Y_obs[control_units[:,None], t]
            Y_pred = X_pred.T.dot(model_lr_vertical.coef_)+ model_lr_vertical.intercept_
            Ythat.append(Y_pred.item())
        Y_hat[idx, :] = Ythat
    return Y_hat
    
def generate_nonlinear_covariate_effect(X, T, hidden_dim=32):
    """
    Generate a covariate effect matrix (N, T) by applying a one-hidden-layer ReLU NN to X.
    This mimics the transformation used in many NN models.
    """
    N, P = X.shape
    effect = np.zeros((N, T))
    for t in range(T):
        # Use Xavier initialization
        W1 = np.random.randn(P, hidden_dim) * np.sqrt(2. / P)
        b1 = np.random.randn(1, hidden_dim)
        W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
        b2 = np.random.randn(1, 1)
        hidden = np.maximum(X.dot(W1) + b1, 0)
        effect[:, t:t+1] = hidden.dot(W2) + b2
    return effect

def generate_nonlinear_covariate_effect_sinusoidal(X, T):
    """
    Generate a covariate effect matrix (N, T) using a sinusoidal transformation.

    For each time step t, a random weight vector and bias are drawn so that:

        effect[i, t] = sin( X[i,:] dot w_t + b_t )

    Parameters:
        X: numpy array of shape (N, P)
        T: number of time periods

    Returns:
        effect: numpy array of shape (N, T)
    """
    N, P = X.shape
    effect = np.zeros((N, T))
    for t in range(T):
        w_t = np.random.randn(P)
        b_t = np.random.randn()
        #effect[:, t] = np.tanh(np.abs(X.dot(w_t)))**0.5 + b_t
        effect[:, t] = np.abs(X.dot(w_t))**0.5 + b_t
        #effect[:, t] = np.log(np.abs(X.dot(w_t) + b_t))
    return effect

def generate_nonlinear_covariate_effect(X, T, hidden_dim=32):
    """
    Generate a covariate effect matrix (N, T) by applying a one-hidden-layer ReLU NN to X.
    This mimics the transformation used in many NN models.
    """
    N, P = X.shape
    effect = np.zeros((N, T))
    for t in range(T):
        # Use Xavier initialization
        W1 = np.random.randn(P, hidden_dim) * np.sqrt(2. / P)
        b1 = np.random.randn(1, hidden_dim)
        W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2. / hidden_dim)
        b2 = np.random.randn(1, 1)
        hidden = np.maximum(X.dot(W1) + b1, 0)
        effect[:, t:t+1] = hidden.dot(W2) + b2
    return effect


# -----------------------------
# Outer Loop over Data-Generating Models
# -----------------------------
model_types = ['linear', 'sine', 'polynomial', 'relu']

all_metrics = {
    mtype: {
       'K_PCA':[]
    }
    for mtype in model_types
}
# RCAutoRec parameters
mr = 0.5
mod = 'NE'
main_hidden_dim = [30, 30]
act_hidden_dim = [20, 20]
rc_epoch = 30
rc_lr = 0.001
rc_wd = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for mtype in model_types:
    print(f"\n--- Data Generating Process: {mtype} ---")
    for sim in range(num_simulations):
        np.random.seed(sim)  # reset seed for reproducibility
        
        # Generate covariate matrix X (N x P)
        X = np.random.randn(N, P)
        # Generate latent factor matrix F (T x r)
        F = np.random.randn(T, r)
        # Generate unit-specific loadings (Lambda: N x r)
        Lambda = np.random.randn(N, r)
        
        # -----------------------------
        # Incorporate Covariate Effect using a Matrix Beta (P x T)
        # -----------------------------
        # Generate a true beta matrix for the covariate effect:
        base_output = np.vstack([generate_unit_output_model(mtype, Lambda[i], F, X[i])
                                 for i in range(N)])
        # Generate nonlinear covariate effect using our similar one-hidden-layer NN structure
        beta_true = np.random.randn(P, T) + 1  # shape (P, T)
        
        cov_effect = generate_nonlinear_covariate_effect_sinusoidal(X, T)
        # Final outcome Y0: base output + nonlinear covariate effect + noise
        Y0 = base_output + cov_effect + np.random.randn(N, T) * noise_std
        
        # Define treatment structure: treat first N/2 units in post-treatment periods
        control_units = np.arange(N // 2)
        treated_units = np.arange(N // 2, N)
        treat_start = T // 2
        pre_periods = np.arange(treat_start)
        post_periods = np.arange(treat_start, T)
        
        np.random.seed(42)
        unit_effects = 12 + 5 * np.random.randn(len(treated_units), 1)
        Y1 = Y0.copy()
        Y1[treated_units[:, None], post_periods] += unit_effects
        
        Y_obs = Y0.copy()
        Y_obs[treated_units[:, None], post_periods] = Y1[treated_units[:, None], post_periods]
        mu_A = np.mean(Y_obs[control_units[:, None], pre_periods])
        mu_B = np.mean(Y_obs[control_units[:, None], post_periods])
        mu_C = np.mean(Y_obs[treated_units[:, None], pre_periods])
        mu_D = np.mean(Y_obs[treated_units[:, None], post_periods])
        mu0 = (mu_A + mu_B + mu_C) / 3
        Y_obs = Y_obs - mu0
        
        # -----------------------------
        # Step 1: Remove Covariate Effect Using Pre-treatment Data
        # -----------------------------
        cov_effect_hat = np.zeros((N, T))
        nn_models = {}
        # For pre-treatment periods: train using all units.
        for t in pre_periods:
            X_train = X.copy()       # shape: (N, P)
            y_train = Y_obs[:, t]    # (N,)
            model_t = train_simple_nn_regressor(X_train, y_train, num_epochs=400, lr=0.001, verbose=False)
            nn_models[t] = model_t
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                preds = model_t(X_tensor).cpu().numpy().flatten()
            cov_effect_hat[:, t] = preds
        # For post-treatment periods: use only control units.
        for t in post_periods:
            X_train = X[control_units, :]
            y_train = Y_obs[control_units, t]
            model_t = train_simple_nn_regressor(X_train, y_train, num_epochs=400, lr=0.001, verbose=False)
            nn_models[t] = model_t
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                preds = model_t(X_tensor).cpu().numpy().flatten()
            cov_effect_hat[:, t] = preds
        
        # Compute residuals after removing the estimated covariate effect.
        Y_residual = Y_obs - cov_effect_hat
        
        # -----------------------------
        # Step 2: Prepare Data for Autoencoder Training on Residuals
        # -----------------------------
        mask = np.ones_like(Y_residual)
        mask[treated_units[:, None], post_periods] = 0  # missing for treated post-treatment
        X_train = Y_residual.copy()
        X_train[mask == 0] = 0
        
        # Transpose for PyTorch: each column corresponds to a time period.
        X_train_torch = torch.tensor(X_train.T, dtype=torch.float)
        residuals_torch = torch.tensor(Y_residual.T, dtype=torch.float)
        mask_torch = torch.tensor(mask.T, dtype=torch.float)
        
        # -----------------------------
        # Step 3: Train Autoencoder Models on Residuals
        # -----------------------------
        # model_disjoint = train_model(DisjointDecoderAE(N, r), X_train_torch, residuals_torch, mask_torch, epochs, lr_model)
        # model_single = train_model(SingleDecoderAE(N, r), X_train_torch, residuals_torch, mask_torch, epochs, lr_model)
        # Y_hat_disjoint_residual = impute_disjoint(model_disjoint, X_train, treated_units, control_units, post_periods)
        # with torch.no_grad():
        #     Y_recon_single = model_single(X_train_torch).T.numpy()
        # Y_hat_single_residual = Y_recon_single[treated_units[:, None], post_periods]

        # # -----------------------------
        # # Step 4: Train RCAutoRec on Residuals
        # # -----------------------------
        # Y_zero_filled = Y_residual.copy()
        # Y_zero_filled[mask == 0] = 0
        # test_matrix = Y_residual.copy()
        # # Remove covariate effect from Y1 for test_matrix:
        # test_matrix[treated_units[:, None], post_periods] = (Y1 - cov_effect_hat)[treated_units[:, None], post_periods]
        # net_rc, rc_rmse, rc_min_rmse, rc_best_epoch = training_RCAutoRec(
        #     mr, mod, [30, 30], [20, 20],
        #     rc_epoch, rc_lr, rc_wd, device, Y_zero_filled, mask, test_matrix)
        # net_rc.eval()
        # with torch.no_grad():
        #     data_tensor = torch.FloatTensor(Y_zero_filled).to(device)
        #     mask_tensor = torch.FloatTensor(mask).to(device)
        #     predictions_rc = net_rc.forward(data_tensor, mask_tensor)
        #     predictions_rc = predictions_rc.cpu().numpy()
        # Y_hat_rc_residual = predictions_rc[treated_units[:, None], post_periods]

        kpca = KernelPCA(n_components=r,kernel='rbf',fit_inverse_transform=True,gamma=0.1)
        Yzero=Y_residual.copy(); Yzero[mask==0]=0
        kpca.fit(Yzero[control_units])
        Zall= kpca.transform(Yzero)
        Yrec= kpca.inverse_transform(Zall)
        Yk= Yrec[treated_units][:,post_periods]

        # # -----------------------------
        # # Step 6: Impute Using Other Methods on Residuals
        # # -----------------------------
        # Y_hat_did_res = did_impute(Y_residual, treated_units, control_units, pre_periods, post_periods)
        # #Y_hat_mcnnm_full_res = soft_impute(Y_residual, mask, lambda_nuclear, tol_nuclear, max_iter_nuclear)
        # #Y_hat_mcnnm_res = Y_hat_mcnnm_full_res[treated_units[:, None], post_periods]
        # M, a, b, tau = ct.MC_NNM_with_suggested_rank(Y_residual, mask, suggest_r = r)
        # Y_hat_mcnnm_res = M + np.tile(a, (1, M.shape[1]))+np.tile(b.T, (M.shape[0],1))
        # Y_hat_mcnnm_res = Y_hat_mcnnm_res[treated_units[:, None], post_periods]
        
        # Y_hat_vr_ols_res = vertical_regression_impute(Y_residual, treated_units, control_units, pre_periods, post_periods)
        # Y_hat_disjoint = Y_hat_disjoint_residual + cov_effect_hat[treated_units][:, post_periods] + mu0
        # Y_hat_single   = Y_hat_single_residual   + cov_effect_hat[treated_units][:, post_periods] + mu0
        # Y_hat_rc       = Y_hat_rc_residual         + cov_effect_hat[treated_units][:, post_periods] + mu0
        # Y_hat_did      = Y_hat_did_res             + cov_effect_hat[treated_units][:, post_periods] + mu0
        # Y_hat_mcnnm    = Y_hat_mcnnm_res           + cov_effect_hat[treated_units][:, post_periods] + mu0
        # Y_hat_vr       = Y_hat_vr_ols_res          + cov_effect_hat[treated_units][:, post_periods] + mu0
        Y_hat_yk       = Yk                        + cov_effect_hat[treated_units][:, post_periods] + mu0
        
        # -----------------------------
        # Step 8: Compute Metrics for This Simulation
        # -----------------------------
        true_Y0 = Y0[treated_units[:, None], post_periods]
        true_Y1 = Y1[treated_units[:, None], post_periods]
        tau_it = unit_effects  # true treatment effect per unit
        

        # m_disjoint = compute_metrics(true_Y0, true_Y1, Y_hat_disjoint, tau_it)
        # m_single   = compute_metrics(true_Y0, true_Y1, Y_hat_single, tau_it)
        # m_rc       = compute_metrics(true_Y0, true_Y1, Y_hat_rc, tau_it)
        # m_did      = compute_metrics(true_Y0, true_Y1, Y_hat_did, tau_it)
        # m_mcnnm    = compute_metrics(true_Y0, true_Y1, Y_hat_mcnnm, tau_it)
        # m_vert     = compute_metrics(true_Y0, true_Y1, Y_hat_vr, tau_it)
        m_kpca     = compute_metrics(true_Y0, true_Y1, Y_hat_yk, tau_it)
        
        #all_metrics[mtype]['Disjoint'].append(m_disjoint)
        # all_metrics[mtype]['Single'].append(m_single)
        # all_metrics[mtype]['RCAutoRec'].append(m_rc)
        # all_metrics[mtype]['DiD'].append(m_did)
        # all_metrics[mtype]['Ver_reg'].append(m_vert)
        # all_metrics[mtype]['MCNNM'].append(m_mcnnm)
        all_metrics[mtype]['K_PCA'].append(m_kpca)
        
    # End simulation loop for current model type
    # avg_disjoint = np.mean(all_metrics[mtype]['Disjoint'], axis=0)
    # avg_single   = np.mean(all_metrics[mtype]['Single'], axis=0)
    # avg_RCAutoRec = np.mean(all_metrics[mtype]['RCAutoRec'], axis=0)
    # avg_did      = np.mean(all_metrics[mtype]['DiD'], axis=0)
    # avg_mcnnm    = np.mean(all_metrics[mtype]['MCNNM'], axis=0)
    # avg_vert     = np.mean(all_metrics[mtype]['Ver_reg'], axis=0)
    avg_KPCA     = np.mean(all_metrics[mtype]['K_PCA'], axis=0)
    
    for key in all_metrics[mtype]:
        avg = np.mean(all_metrics[mtype][key], axis=0)
        se  = compute_se(all_metrics[mtype][key])

        print(f"{key:10s} | MAE_comp = {avg[1]:.3f} ± {se[1]:.3f}   MSE_comp = {avg[3]:.3f} ± {se[3]:.3f} \n\t\t MAE_unit = {avg[5]:.3f} ± {se[5]:.3f}    MSE_unit = {avg[6]:.3f} ± {se[6]:.3f} \n\t\t MAE_ave = {avg[7]:.4f} ± {se[7]:.4f}  MSE_ave= {avg[8]:.4f} ± {se[8]:.4f}  ")
