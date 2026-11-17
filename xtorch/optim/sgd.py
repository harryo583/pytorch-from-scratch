
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.velocity = [None for _ in self.params]

    def step(self):
        raise NotImplementedError
