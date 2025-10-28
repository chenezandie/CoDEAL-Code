import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from copy import deepcopy
import causaltensor as ct

from torch.nn.modules.activation import ReLU

#############################################
############ code of AEMC-NE ################
#############################################
# The code of AEMC-NE is cited from: https://openreview.net/attachment?id=kPrxk6tUcg&name=supplementary_material
from sklearn.decomposition import KernelPCA




def kpca_impute(Y_obs, treated_units, control_units, pre_periods, post_periods):
    kpca = KernelPCA(n_components=r,kernel='rbf',fit_inverse_transform=True,gamma=0.1)
    Yzero=Y_res.copy(); Yzero[mask==0]=0
    kpca.fit(Yzero[control_units])
    Zall= kpca.transform(Yzero)
    Yrec= kpca.inverse_transform(Zall)
    Yk= Yrec[treated_units][:,post_periods]
    return Yk


def impute_kpca(i, j, Y_res, mask, cov_hat, mu0):
    r0, r1 = i*n_group, (i+1)*n_group
    c0, c1 = j*t_group, (j+1)*t_group
    re, ce = r1, c1
    
    Y_sub = Y_res[:re, :ce].copy()
    k1 = k-j
    k2 = k-i
    control_units = np.arange(k1*n_group)
    treated_units = np.arange(k1*n_group, re)
    pre_periods  = np.arange(k2*t_group)
    post_periods = np.arange(k2*t_group, ce)
    M_sub = mask[:re, :ce].copy()
    M_sub[treated_units[:, None], post_periods] = 0

    block = kpca_impute(Y_sub, treated_units, control_units, pre_periods, post_periods)
    nrow_block, ncol_block = block.shape
    return block[(nrow_block-n_group):nrow_block,(ncol_block-t_group):ncol_block] + cov_hat[r0:r1, c0:c1] + mu0


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


def impute_verreg(i, j, Y_res, mask, cov_hat, mu0):
    r0, r1 = i*n_group, (i+1)*n_group
    c0, c1 = j*t_group, (j+1)*t_group
    re, ce = r1, c1

    Y_sub = Y_res[:re, :ce].copy()
    k1 = k-j
    k2 = k-i
    control_units = np.arange(k1*n_group)
    treated_units = np.arange(k1*n_group, re)
    pre_periods  = np.arange(k2*t_group)
    post_periods = np.arange(k2*t_group, ce)
    M_sub = mask[:re, :ce].copy()
    M_sub[treated_units[:, None], post_periods] = 0
    
    block = vertical_regression_impute(Y_sub, treated_units, control_units, pre_periods, post_periods)
    nrow_block, ncol_block = block.shape
    return block[(nrow_block-n_group):nrow_block,(ncol_block-t_group):ncol_block] + cov_hat[r0:r1, c0:c1] + mu0

# -----------------------------
# 0. Setup
# -----------------------------
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)

# -----------------------------
# 1. Simulation Settings
# -----------------------------
N, T, r, P = 100, 200, 4, 3       # units, time periods, latent dim, covariates
k = 10                             # number of staggered cohorts
n_group, t_group = N//k, T//k
noise_std = 0.5
epochs = 50
lr_model = 2e-3

# -----------------------------
# 2. Data‐Generating Helpers
# -----------------------------
def h_linear(Lambda_i, F_t, C):
    return C * np.dot(Lambda_i, F_t)

def h_sine(Lambda_i, F_t, C):
    return C * np.sin(np.dot(Lambda_i, F_t))

def h_polynomial(Lambda_i, F_t, C1, C2):
    dot_val = np.dot(Lambda_i, F_t)
    return C1 * dot_val + C2 * (dot_val ** 2)

def generate_unit_output(F_rt, X_i, hidden_dim=10):
    # F_rt: (r, T)
    W1 = np.random.randn(hidden_dim, F_rt.shape[0]) * 0.5
    b1 = np.random.randn(hidden_dim, 1) * 0.5
    W2 = np.random.randn(1, hidden_dim) * 0.5
    b2 = np.random.randn(1, 1) * 0.5
    H = np.maximum(W1 @ F_rt + b1, 0)  # (hidden_dim, T)
    return (W2 @ H + b2).flatten()    # (T,)

def generate_unit_output_model(model_type, Lambda_i, F, X_i):
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
        C1, C2 = np.random.randn()*0.2, np.random.randn()*0.2
        for t in range(T_local):
            y[t] = h_polynomial(Lambda_i, F[t], C1, C2)
    elif model_type == 'relu':
        # pass F.T so it’s (r, T)
        y = generate_unit_output(F.T, X_i, hidden_dim=10)
    else:
        raise ValueError("Invalid model type")
    return y

# -----------------------------
# 3. Autoencoder Definitions
# -----------------------------
class DisjointDecoderAE(nn.Module):
    def __init__(self, num_units, latent_dim, enc_hidden=[64,64,64], dec_hidden=[64,64,64]):
        super().__init__()
        layers, dim = [], num_units
        for h in enc_hidden:
            layers += [nn.Linear(dim,h), nn.ReLU()]; dim = h
        layers += [nn.Linear(dim, latent_dim)]
        self.encoder = nn.Sequential(*layers)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, dec_hidden[0]), nn.ReLU(),
                nn.Linear(dec_hidden[0], dec_hidden[1]), nn.ReLU(),
                nn.Linear(dec_hidden[1], 1)
            ) for _ in range(num_units)
        ])
    def forward(self, x):
        z = self.encoder(x)
        return torch.cat([d(z) for d in self.decoders], dim=1)

class SingleDecoderAE(nn.Module):
    def __init__(self, num_units, latent_dim, enc_hidden=[16,16,16], dec_hidden=[16,16,16]):
        super().__init__()
        layers, dim = [], num_units
        for h in enc_hidden:
            layers += [nn.Linear(dim,h), nn.ReLU()]; dim = h
        layers += [nn.Linear(dim, latent_dim)]
        self.encoder = nn.Sequential(*layers)
        layers, dim = [], latent_dim
        for h in dec_hidden:
            layers += [nn.Linear(dim,h), nn.ReLU()]; dim = h
        layers += [nn.Linear(dim, num_units)]
        self.decoder = nn.Sequential(*layers)
    def forward(self, x):
        return self.decoder(self.encoder(x))

# -----------------------------
# 4. Train & Impute Helpers
# -----------------------------
def train_model(model, X, Y, mask, epochs=200, lr=5e-4):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(X)
        loss = (loss_fn(pred, Y) * mask).sum() / mask.sum()
        loss.backward(); opt.step()
    return deepcopy(model.eval())

def impute_disjoint(model, X_sub, M_sub, units, cols):
    model.eval()
    N_sub, T_sub = X_sub.shape
    with torch.no_grad():
        Z = []
        for t in cols:
            vec = np.zeros(N_sub)
            obs = np.where(M_sub[:,t] > 0)[0]
            vec[obs] = X_sub[obs, t]
            zt = model.encoder(torch.tensor(vec, dtype=torch.float).unsqueeze(0))
            Z.append(zt)
        Z = torch.cat(Z, dim=0)
        out = [model.decoders[u](Z).squeeze(-1).cpu().numpy() for u in units]
    return np.stack(out, axis=0)


def impute_four_block_disjoint(i, j, Y_res, mask, cov_hat, mu0):
    r0, r1 = i*n_group, (i+1)*n_group
    c0, c1 = j*t_group, (j+1)*t_group
    re, ce = r1, c1

    Y_sub = Y_res[:re, :ce].copy()
    k1 = k-j
    k2 = k-i
    control_units = np.arange(k1*n_group)
    treated_units = np.arange(k1*n_group, re)
    pre_periods  = np.arange(k2*t_group)
    post_periods = np.arange(k2*t_group, ce)
    M_sub = mask[:re, :ce].copy()
    M_sub[treated_units[:, None], post_periods] = 0
    
    X_tr = torch.tensor(Y_sub.T, dtype=torch.float)
    Y_tr = torch.tensor(Y_sub.T, dtype=torch.float)
    M_tr = torch.tensor(M_sub.T, dtype=torch.float)

    ae = DisjointDecoderAE(re, r)
    ae = train_model(ae, X_tr, Y_tr, M_tr, epochs, lr_model)

    block = impute_disjoint(ae, Y_sub, M_sub, np.arange(r0,r1), np.arange(c0,c1))
    return block + cov_hat[r0:r1, c0:c1] + mu0

def impute_four_block_single(i, j, Y_res, mask, cov_hat, mu0):
    r0, r1 = i*n_group, (i+1)*n_group
    c0, c1 = j*t_group, (j+1)*t_group
    re, ce = r1, c1
    
    Y_sub = Y_res[:re, :ce].copy()
    k1 = k-j
    k2 = k-i
    control_units = np.arange(k1*n_group)
    treated_units = np.arange(k1*n_group, re)
    pre_periods  = np.arange(k2*t_group)
    post_periods = np.arange(k2*t_group, ce)
    M_sub = mask[:re, :ce].copy()
    M_sub[treated_units[:, None], post_periods] = 0

    X_tr = torch.tensor(Y_sub.T, dtype=torch.float)
    Y_tr = torch.tensor(Y_sub.T, dtype=torch.float)
    M_tr = torch.tensor(M_sub.T, dtype=torch.float)

    ae = SingleDecoderAE(re, r)
    ae = train_model(ae, X_tr, Y_tr, M_tr, epochs, lr_model)

    with torch.no_grad():
        pred = ae(X_tr).cpu().numpy().T  # (re, ce)
    block = pred[r0:r1, c0:c1]
    return block + cov_hat[r0:r1, c0:c1] + mu0


def did_impute(Y_obs, treated_units, control_units, pre_periods, post_periods):
    pre_baseline =  np.mean(Y_obs[control_units[:, None], pre_periods], axis=1)
    trend = np.array([np.mean(Y_obs[control_units, t] - pre_baseline) for t in post_periods])
    baseline_treated = np.mean(Y_obs[treated_units[:, None], pre_periods], axis=1)
    Y_hat_did = np.outer(baseline_treated, np.ones(len(post_periods))) + trend
    return Y_hat_did


def impute_did(i, j, Y_res, mask, cov_hat, mu0):
    r0, r1 = i*n_group, (i+1)*n_group
    c0, c1 = j*t_group, (j+1)*t_group
    re, ce = r1, c1
    
    Y_sub = Y_res[:re, :ce].copy()
    k1 = k-j
    k2 = k-i
    control_units = np.arange(k1*n_group)
    treated_units = np.arange(k1*n_group, re)
    pre_periods  = np.arange(k2*t_group)
    post_periods = np.arange(k2*t_group, ce)
    M_sub = mask[:re, :ce].copy()
    M_sub[treated_units[:, None], post_periods] = 0

    block = did_impute(Y_sub, treated_units, control_units, pre_periods, post_periods)
    nrow_block, ncol_block = block.shape
    return block[(nrow_block-n_group):nrow_block,(ncol_block-t_group):ncol_block] + cov_hat[r0:r1, c0:c1] + mu0



# -----------------------------
# 5. Main Pipeline
# -----------------------------
# generate covariates & factors
X         = np.random.randn(N, P)
F         = np.random.randn(T, r)
Lambda    = np.random.randn(N, r)
beta_true = np.random.randn(P, T) + 1


#control_units = np.arange(N // 5)
#treated_units = np.arange(N // 5, 2*N // 5)
#pre_periods = np.arange(T // 5) 
#post_periods = np.arange(T // 5, 2*T // 5)

# baseline Y0 (control potential outcomes)
Y0 = np.vstack([
    generate_unit_output_model('linear', Lambda[i], F, X[i])
    for i in range(N)
])
#Y0 = Y0 + X.dot(beta_true) + np.random.randn(N, T) * noise_std
Y0 = Y0  + np.random.randn(N, T) * noise_std

# staggered treatment assignment
groups       = np.repeat(np.arange(k), n_group)
treat_starts = T - groups * t_group

# apply treatment effects
unit_eff = 12 + 5 * np.random.randn(N)
Y1 = Y0.copy()
for i in range(N):
    ts = treat_starts[i]
    if ts < T:
        Y1[i, ts:] += unit_eff[i]

# observed outcomes
Y_obs = Y0.copy()
for i in range(N):
    ts = treat_starts[i]
    if ts < T:
        Y_obs[i, ts:] = Y1[i, ts:]

# center on never-treated (group 0)
ctrl0 = np.where(treat_starts >= T)[0]
mu0   = Y_obs[ctrl0].mean()
Y_obs -= mu0

# remove covariate effect via OLS
beta_est = np.zeros((P, T))
for t in range(T):
    ctrl = np.where(treat_starts > t)[0]
    beta_est[:, t] = LinearRegression().fit(X[ctrl], Y_obs[ctrl, t]).coef_
cov_hat = X.dot(beta_est)
# Y_res   = Y_obs - cov_hat

## if not considering covariate effect:
cov_hat = np.zeros((N, T))
Y_res   = Y_obs - cov_hat

# mask post-treatment entries
mask = np.ones_like(Y_res)
for i in range(N):
    ts = treat_starts[i]
    if ts < T:
        mask[i, ts:] = 0

# initialize global imputed matrices
Y_imp_disjoint = Y_obs.copy()
Y_imp_single   = Y_obs.copy()
Y_imp_did      = Y_obs.copy()
Y_imp_ver      = Y_obs.copy()
Y_imp_RCAutoRec= Y_obs.copy()
Y_imp_MCNNM    = Y_obs.copy()

mr = 0.5
mod = 'NE'
main_hidden_dim = [30, 30]
act_hidden_dim = [20, 20]
rc_epoch = 30
rc_lr = 0.001
rc_wd = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_types = ['linear', 'sine', 'polynomial', 'relu']


mae_metrics = {
    mtype: {
        'Disjoint': [], 'Single': [], 'RCAutoRec': [],
        'DiD': [], 'MCNNM': [], 'Ver_reg': [], 'Kpca': []}
    for mtype in model_types
}

mse_metrics = {
    mtype: {
        'Disjoint': [], 'Single': [], 'RCAutoRec': [],
        'DiD': [], 'MCNNM': [], 'Ver_reg': [], 'Kpca': []}
    for mtype in model_types
}


num_simulations = 50

for mtype in model_types:
    print(f"\n--- Data Generating Process: {mtype} ---")
    for sim in range(num_simulations):
        X         = np.random.randn(N, P)
        F         = np.random.randn(T, r)
        Lambda    = np.random.randn(N, r)
        beta_true = np.random.randn(P, T) + 1
        #control_units = np.arange(N // 5)
        #treated_units = np.arange(N // 5, 2*N // 5)
        #pre_periods = np.arange(T // 5) 
        #post_periods = np.arange(T // 5, 2*T // 5)
        Y0 = np.vstack([generate_unit_output_model(mtype, Lambda[i], F, X[i])
                        for i in range(N)
                       ])
        Y0 = Y0  + np.random.randn(N, T) * noise_std
        groups       = np.repeat(np.arange(k), n_group)
        treat_starts = T - groups * t_group
        
        unit_eff = 12 + 5 * np.random.randn(N)
        Y1 = Y0.copy()
        
        for i in range(N):
            ts = treat_starts[i]
            if ts < T:
                Y1[i, ts:] += unit_eff[i]
        Y_obs = Y0.copy()
        for i in range(N):
            ts = treat_starts[i]
            if ts < T:
                Y_obs[i, ts:] = Y1[i, ts:]
                
        ctrl0 = np.where(treat_starts >= T)[0]
        mu0   = Y_obs[ctrl0].mean()
        Y_obs -= mu0
        # beta_est = np.zeros((P, T))
        # for t in range(T):
        #     ctrl = np.where(treat_starts > t)[0]
        #     beta_est[:, t] = LinearRegression().fit(X[ctrl], Y_obs[ctrl, t]).coef_
        # cov_hat = X.dot(beta_est)
        Y_res   = Y_obs
        mask = np.ones_like(Y_res)
        for i in range(N):
            ts = treat_starts[i]
            if ts < T:
                mask[i, ts:] = 0
        Y_imp_disjoint = Y_obs.copy()
        Y_imp_single   = Y_obs.copy()
        Y_imp_did      = Y_obs.copy()
        Y_imp_ver      = Y_obs.copy()
        Y_imp_RCAutoRec= Y_obs.copy()
        Y_imp_MCNNM    = Y_obs.copy()
        Y_imp_kpca = Y_obs.copy()
        for i in range(1, k):
            for j in range(k - i, k):
                print(f"Block (row grp={i}, col grp={j})")
                r0, r1 = i*n_group, (i+1)*n_group
                c0, c1 = j*t_group, (j+1)*t_group
                re, ce = r1, c1
                
                Y_sub = Y_res[:re, :ce].copy()
                k1 = k-j
                k2 = k-i
                control_units = np.arange(k1*n_group)
                treated_units = np.arange(k1*n_group, re)
                pre_periods  = np.arange(k2*t_group)
                post_periods = np.arange(k2*t_group, ce)
                M_sub = mask[:re, :ce].copy()
                M_sub[treated_units[:, None], post_periods] = 0
                
                Y_zero_filled = Y_sub.copy()
                Y_zero_filled[M_sub == 0] = 0
                test_matrix = Y_sub.copy()
                test_matrix[treated_units[:, None], post_periods] = Y_sub[treated_units[:, None], post_periods]
                
                ## RCAutoRec
                net_rc, rc_rmse, rc_min_rmse, rc_best_epoch = training_RCAutoRec(mr, mod, [30, 30], [20, 20],
                                                                                 rc_epoch, rc_lr, rc_wd, device, Y_zero_filled, M_sub, test_matrix)
                net_rc.eval()
                with torch.no_grad():
                    data_tensor = torch.FloatTensor(Y_zero_filled).to(device)
                    mask_tensor = torch.FloatTensor(M_sub).to(device)
                    predictions_rc = net_rc.forward(data_tensor, mask_tensor)
                    predictions_rc = predictions_rc.cpu().numpy()
                Y_hat_rc_residual = predictions_rc[treated_units[:, None], post_periods]
                nrow_block, ncol_block = Y_hat_rc_residual.shape
                blk_RCAutoRec = Y_hat_rc_residual[(nrow_block-n_group):nrow_block,(ncol_block-t_group):ncol_block]+ cov_hat[r0:r1, c0:c1] + mu0

                
                ## Disjoint AE
                blk_d = impute_four_block_disjoint(i, j, Y_res, mask, cov_hat, mu0)
                
                 ## Single AE
                blk_s = impute_four_block_single(i, j, Y_res, mask, cov_hat, mu0)
                
                ## DID
                blk_did = impute_did(i, j, Y_res, mask, cov_hat, mu0)
                
                ## Vertical Regression
                blk_ver = impute_verreg(i, j, Y_res, mask, cov_hat, mu0)
                tau_it = unit_eff # true treatment effect per unit
                
                ## MCNNM
                M, a, b, tau = ct.MC_NNM_with_suggested_rank(Y_sub, M_sub, suggest_r = r)
                Y_hat_mcnnm_res = M + np.tile(a, (1, M.shape[1]))+np.tile(b.T, (M.shape[0],1))
                Y_hat_mcnnm_res = Y_hat_mcnnm_res[treated_units[:, None], post_periods]
                Y_hat_mcnnm2 = Y_hat_mcnnm_res
                #blk_MCNNM = Y_hat_mcnnm2
                nrow_block, ncol_block = Y_hat_mcnnm2.shape
                blk_MCNNM = Y_hat_mcnnm2[(nrow_block-n_group):nrow_block,(ncol_block-t_group):ncol_block] + cov_hat[r0:r1, c0:c1] + mu0

                blk_kpca = impute_kpca(i, j, Y_res, mask, cov_hat, mu0)

                Y_imp_kpca[r0:r1, c0:c1] = blk_kpca
                r0, r1 = i*n_group, (i+1)*n_group
                c0, c1 = j*t_group, (j+1)*t_group
                Y_imp_disjoint[r0:r1, c0:c1] = blk_d
                Y_imp_single[  r0:r1, c0:c1] = blk_s
                Y_imp_did[  r0:r1, c0:c1] = blk_did
                Y_imp_ver[  r0:r1, c0:c1] = blk_ver
                Y_imp_RCAutoRec[  r0:r1, c0:c1] = blk_RCAutoRec
                Y_imp_MCNNM[  r0:r1, c0:c1] = blk_MCNNM
        missing = (mask == 0)
        mse_d = ((Y_imp_disjoint[missing] - Y0[missing])**2).mean()
        mae_d =  np.abs(Y_imp_disjoint[missing] - Y0[missing]).mean()
        mse_s = ((Y_imp_single[missing]   - Y0[missing])**2).mean()
        mae_s =  np.abs(Y_imp_single[missing]   - Y0[missing]).mean()
        mse_did = ((Y_imp_did[missing]   - Y0[missing])**2).mean()
        mae_did =  np.abs(Y_imp_did[missing]   - Y0[missing]).mean()
        mse_ver = ((Y_imp_ver[missing]   - Y0[missing])**2).mean()
        mae_ver =  np.abs(Y_imp_ver[missing]   - Y0[missing]).mean()
        mse_RCAutoRec = ((Y_imp_RCAutoRec[missing]   - Y0[missing])**2).mean()
        mae_RCAutoRec =  np.abs(Y_imp_RCAutoRec[missing]   - Y0[missing]).mean()
        mse_MCNNM = ((Y_imp_MCNNM[missing]   - Y0[missing])**2).mean()
        mae_MCNNM =  np.abs(Y_imp_MCNNM[missing]   - Y0[missing]).mean()
        mae_metrics[mtype]['Disjoint'].append(mae_d)
        mae_metrics[mtype]['Single'].append(mae_s)
        mae_metrics[mtype]['RCAutoRec'].append(mae_RCAutoRec)
        mae_metrics[mtype]['DiD'].append(mae_did)
        mae_metrics[mtype]['MCNNM'].append(mae_MCNNM)
        mae_metrics[mtype]['Ver_reg'].append(mae_ver)
        mse_metrics[mtype]['Disjoint'].append(mse_d)
        mse_metrics[mtype]['Single'].append(mse_s)
        mse_metrics[mtype]['RCAutoRec'].append(mse_RCAutoRec)
        mse_metrics[mtype]['DiD'].append(mse_did)
        mse_metrics[mtype]['MCNNM'].append(mse_MCNNM)
        mse_metrics[mtype]['Ver_reg'].append(mse_ver)
        mse_dk = ((Y_imp_kpca[missing] - Y0[missing])**2).mean()
        mae_dk =  np.abs(Y_imp_kpca[missing] - Y0[missing]).mean()
        mae_metrics[mtype]['Kpca'].append(mae_dk)
        mse_metrics[mtype]['Kpca'].append(mse_dk)


def compute_se(metric_list):
    """
    metric_list: list of tuples (est_ATE, MAE_comp, MAE_treat, MSE_comp, MSE_treat)
    returns: array of SEs for each metric
    """
    arr = np.array(metric_list)  # shape (S,5)
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return se

for mtype in model_types:
    print(f"\n--- Data Generating Process: {mtype} ---")
    for key in mae_metrics[mtype]:
        avg = np.mean(mse_metrics[mtype][key], axis=0)
        se  = compute_se(mse_metrics[mtype][key])
        avg1 = np.mean(mae_metrics[mtype][key], axis=0)
        se1  = compute_se(mae_metrics[mtype][key])
        print(key + ' MAE')
        print(avg1)
        print('MAE standard error')
        print(se1)
        print(key + ' MSE')
        print(avg)
        print('MSE standard error')
        print(se)




