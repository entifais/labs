from os import altsep
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import torch
from sklearn.base import RegressorMixin
from src.poly import PolinomialFunction

def bias_variance_experiment(
    oracle: PolinomialFunction,
    model: RegressorMixin,
    num_train_datasets: int = 100, 
    num_train_examples_per_dataset: int = 25,
    x_test: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_test is None:
        x_test = torch.arange(-2, 2, 0.01)

    y_test = torch.zeros(x_test.size(0), num_train_datasets)
    y_real = oracle(x_test, add_noise=None)

    for i in range(num_train_datasets):
        x_train = torch.rand(num_train_examples_per_dataset) * 2 - 1
        y_train = oracle(x_train)

        model = model.fit(x_train, y_train)
        y_test[:, i] = model.predict(x_test).squeeze()

    return x_test, y_real, y_test

def plot_bias_variance(x_test, y_real, y_test, **kwargs):
    x = x_test.numpy()
    mean = y_test.mean(1).numpy()
    std = y_test.std(1).numpy()

    _, ax = plt.subplots(**kwargs)

    ax.plot(x, mean)
    ax.plot(x, y_real.numpy(), 'r--', alpha=0.3)
    ax.fill_between(x, mean-std, mean+std ,alpha=0.3)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax