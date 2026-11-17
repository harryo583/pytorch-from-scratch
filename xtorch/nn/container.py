from .module import Module

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
