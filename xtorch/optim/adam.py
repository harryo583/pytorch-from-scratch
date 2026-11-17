
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
        raise NotImplementedError
