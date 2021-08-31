import math
import matplotlib.pyplot as plt

import torch

def pol_value(alpha, x):
    x_pow = x.view(-1, 1) ** torch.arange(alpha.size(0)).view(1, -1)
    return x_pow @ alpha

def fit_alpha(x, y, D, a = 0, b = 1, rho = 1e-12):
    M = x.view(-1, 1) ** torch.arange(D + 1).view(1, -1)
    B = y

    if D >= 2:
        q = torch.arange(2, D + 1, dtype = x.dtype).view(1, -1)
        r = q.view(-1,  1)
        beta = x.new_zeros(D + 1, D + 1)
        beta[2:, 2:] = (q-1) * q * (r-1) * r * (b**(q+r-3) - a**(q+r-3))/(q+r-3)
        l, U = beta.eig(eigenvectors = True)
        Q = U @ torch.diag(l[:, 0].clamp(min = 0) ** 0.5) # clamp deals with ~0 negative values
        B = torch.cat((B, y.new_zeros(Q.size(0))), 0)
        M = torch.cat((M, math.sqrt(rho) * Q.t()), 0)

    return torch.lstsq(B, M).solution[:D+1, 0]

def phi(x):
    # The "ground truth"
    return torch.abs(torch.abs(x - 0.4) - 0.2) + x/2 - 0.1

def compute_mse(nb_train_samples=8, nb_runs=250, D_max=16, train_noise_std=0.):
    mse_train = torch.zeros(nb_runs, D_max + 1)
    mse_test = torch.zeros(nb_runs, D_max + 1)

    for k in range(nb_runs):
        x_train = torch.rand(nb_train_samples, dtype = torch.float64)
        y_train = phi(x_train)
        if train_noise_std > 0:
            y_train = y_train + torch.empty_like(y_train).normal_(0, train_noise_std)
        x_test = torch.linspace(0, 1, 100, dtype = x_train.dtype)
        y_test = phi(x_test)

        for D in range(D_max + 1):
            alpha = fit_alpha(x_train, y_train, D)
            mse_train[k, D] = ((pol_value(alpha, x_train) - y_train)**2).mean()
            mse_test[k, D] = ((pol_value(alpha, x_test) - y_test)**2).mean()

    return mse_train.median(0).values, mse_test.median(0).values

def plot_example(D, train_noise_std = 0, nb_train_samples=8):
    torch.manual_seed(9) # I picked that for pretty

    x_train = torch.rand(nb_train_samples, dtype = torch.float64)
    y_train = phi(x_train)
    if train_noise_std > 0:
        y_train = y_train + torch.empty_like(y_train).normal_(0, train_noise_std)
    x_test = torch.linspace(0, 1, 100, dtype = x_train.dtype)
    y_test = phi(x_test)

    fig = plt.figure(dpi=244)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Degree {D}')
    ax.set_ylim(-0.1, 1.1)
    ax.plot(x_test, y_test, color = 'black', label = 'Test values')
    ax.scatter(x_train, y_train, color = 'blue', label = 'Train samples')

    alpha = fit_alpha(x_train, y_train, D)
    ax.plot(x_test, pol_value(alpha, x_test), color = 'red', label = 'Fitted polynomial')

    ax.legend(frameon = False)

def plot_mse_degree(nb_train_samples=8, D_max= 16):
    fig = plt.figure(dpi=244)

    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel('Polynomial degree', labelpad = 10)
    ax.set_ylabel('MSE', labelpad = 10)

    ax.axvline(x = nb_train_samples - 1,
            color = 'gray', linewidth = 0.5, linestyle = '--')

    ax.text(nb_train_samples - 1.2, 1e-4, 'Nb. params = nb. samples',
            fontsize = 10, color = 'gray',
            rotation = 90, rotation_mode='anchor')

    mse_train, mse_test = compute_mse(nb_train_samples)
    ax.plot(torch.arange(D_max + 1), mse_train, color = 'blue', label = 'Train')
    ax.plot(torch.arange(D_max + 1), mse_test, color = 'red', label = 'Test')

    ax.legend(frameon = False)
