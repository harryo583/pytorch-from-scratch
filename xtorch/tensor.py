import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad

        self.grad = None

        # build autograd graph
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return (f"Tensor({self.data}, requires_grad={self.requires_grad})")

    # Aithmetic

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        raise NotImplementedError

    def __sub__(self, other):
        return self + (-other)

    def __matmul__(self, other):
        raise NotImplementedError

    def __pow__(self, exponent):
        raise NotImplementedError

    # ---------- Reductions ----------

    def sum(self, axis=None, keepdims=False):
        raise NotImplementedError

    def mean(self, axis=None, keepdims=False):
        raise NotImplementedError

    # ---------- Shape operations ----------

    def reshape(self, *shape):
        raise NotImplementedError

    def transpose(self, *axes):
        raise NotImplementedError

    @property
    def T(self):
        return self.transpose()

    # ---------- Activations ----------

    def relu(self):
        raise NotImplementedError

    def exp(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    # ---------- Autograd ----------

    def backward(self, grad=None):
        raise NotImplementedError

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return Tensor(self.data.copy())
