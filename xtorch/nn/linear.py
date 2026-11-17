import numpy as np

from .module import Module
from .parameter import Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
        self.bias = (Parameter(np.zeros(out_features)) if bias else None)

    def forward(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y
