from sklearn.base import RegressorMixin
import torch

class Oracle:
    def noise(self, size: int) -> torch.Tensor:
        return torch.randn((size, 1)) * self.noise_std

    def __call__(self, x, add_noise) -> torch.Tensor:
        raise NotImplementedError

class PolinomialFunction(Oracle):
    def __init__(self, order, noise_std=0.5):
        self.order = order + 1
        self.noise_std = noise_std
        self.alpha = torch.randn(size=(self.order, 1)).float() * 10

    def __call__(self, x, add_noise=True):
        x_pow = x.view(-1, 1) ** torch.arange(self.order)
        res = (x_pow @ self.alpha)
        if add_noise:
            res += self.noise(x.size(0))
        return res

class PolynomialRegression(RegressorMixin):
    def __init__(self, order):
        self.order = order + 1
        self.alpha = torch.rand(self.order)

    def predict(self, x):
        x_pow = x.view(-1, 1) ** torch.arange(self.order).view(1, -1)
        return x_pow @ self.alpha

    def fit(self, X, y):
        x_pow = X.view(-1, 1) ** torch.arange(self.order).view(1, -1)

        self.alpha = torch.linalg.lstsq(x_pow, y).solution
        return self