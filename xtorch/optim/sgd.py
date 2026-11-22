from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [None for _ in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            velocity = self.velocity[i]
            velocity = param.grad.copy() if velocity is None else self.momentum * velocity + param.grad
            self.velocity[i] = velocity
            param.data -= self.lr * velocity