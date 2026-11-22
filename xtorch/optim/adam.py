import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.t = 0

        self.m = [None for _ in self.params]
        self.v = [None for _ in self.params]

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad.copy()
            m = self.m[i]
            m = (1 - self.beta1) * grad if m is None else self.beta1 * m + (1 - self.beta1) * grad
            self.m[i] = m
            
            v = self.v[i]
            v = (1 - self.beta2) * grad ** 2 if v is None else self.beta2 * v + (1 - self.beta2) * grad ** 2
            self.v[i] = v
            
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)