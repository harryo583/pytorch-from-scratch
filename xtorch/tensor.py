import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None

        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        # TODO
        raise NotImplementedError

    def __mul__(self, other):
        # TODO
        raise NotImplementedError

    def __matmul__(self, other):
        # TODO
        raise NotImplementedError

    def backward(self, grad=None):
        # TODO
        raise NotImplementedError