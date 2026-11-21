import numpy as np
from .autograd import *

class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None

        # build autograd graph
        self._prev = tuple(_parents)  # parents come before children in a forward pass
        self._op = _op
        self._backward = lambda: None  # _backward() -> None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return (f"Tensor({self.data}, requires_grad={self.requires_grad})")

    # Aithmetic

    def __add__(self, other) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)

        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op='+',
        )
        
        def _backward():
            if self.requires_grad:
                grad = unbroadcast(out.grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = unbroadcast(out.grad, other.shape)
                other.grad = grad if other.grad is None else other.grad + grad
        
        out._backward = _backward
        
        return out

    def __radd__(self, other) -> Tensor:
        return self + other

    def __mul__(self, other) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
            _op='*',
        )
        
        def _backward():
            if out.grad is None:
                raise RuntimeError("Output gradient has not been initialized")
            if self.requires_grad:
                grad = unbroadcast(out.grad * other.data, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
            if other.requires_grad:
                grad = unbroadcast(out.grad * self.data, other.shape)
                other.grad = grad if other.grad is None else other.grad + grad
        
        out._backward = _backward
        
        return out

    def __rmul__(self, other) -> Tensor:
        return self * other

    def __neg__(self) -> Tensor:
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="neg",
        )
        
        def _backward():
            if self.requires_grad:
                if out.grad is None:
                    raise RuntimeError("Output gradient has not been initialized")
                grad = -out.grad
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        
        return out

    def __sub__(self, other) -> Tensor:
        return self + (-other)
    
    def __rsub__(self, other) -> Tensor:
        return Tensor(other) - self

    def __matmul__(self, other) -> Tensor:
        raise NotImplementedError
    
    def __pow__(self, exponent) -> "Tensor":
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent)

        out = Tensor(
            self.data ** exponent.data,
            requires_grad=self.requires_grad or exponent.requires_grad,
            _parents=(self, exponent),
            _op="pow",
        )

        def _backward():
            if out.grad is None:
                raise RuntimeError("Output gradient has not been initialized")
            if self.requires_grad:
                grad = out.grad * exponent.data * self.data ** (exponent.data - 1)
                grad = unbroadcast(grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
            if exponent.requires_grad:
                grad = out.grad * out.data * np.log(self.data)
                grad = unbroadcast(grad, exponent.shape)
                exponent.grad = grad if exponent.grad is None else exponent.grad + grad

        out._backward = _backward
        return out
    
    
    def __rpow__(self, base) -> Tensor:
        return Tensor(base) ** self


    def __po1w__(self, exponent) -> Tensor:
        if not isinstance(exponent, Tensor):
            exponent = Tensor(exponent)
        
        out = Tensor(
            self.data ** exponent.data,
            requires_grad=self.requires_grad or exponent.requires_grad,
            _parents=(self, exponent),
            _op="pow",
        )
        
        def _backward():
            if self.requires_grad:
                if out.grad is None:
                    raise RuntimeError("Output gradient has not been initialized")
                grad = out.grad * exponent.data * self.data ** (exponent.data - 1)
                grad = unbroadcast(grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
            
            if exponent.requires_grad:
                pass
        
        out._backward = _backward
                
        return out

    # ---------- Reductions ----------

    def sum(self, axis=None, keepdims=False) -> Tensor:
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _parents=(self,),
            _op="sum",
        )
        
        def _backward():
            if self.requires_grad:
                if out.grad is None:
                    raise RuntimeError("Output gradient has not been initialized")
                grad = unbroadcast(out.grad, self.data.shape)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        
        return out
    
    
    def mean(self, axis=None, keepdims=False) -> Tensor:
        if axis is None:
            count = self.data.size
        elif isinstance(axis, tuple):
            count = np.prod([self.shape[i] for i in axis])
        else:
            count = self.shape[axis]
        
        return self.sum(axis=axis, keepdims=keepdims) * (1 / count)


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
        if grad is None:
            if self.data.size != 1:  # total number of elements > 1
                raise ValueError("Calling backward() on non-scalar requires an explicit gradient")
            grad = np.ones_like(self.data)
        else:
            if isinstance(grad, Tensor):
                grad = grad.data
            grad = np.asarray(grad, dtype=np.float64)

        self.grad = grad
        topological_order = topological_sort(self)
        for node in reversed(topological_order):
            node._backward()
    

    def zero_grad(self):
        self.grad = None

    def detach(self):
        return Tensor(self.data.copy())
