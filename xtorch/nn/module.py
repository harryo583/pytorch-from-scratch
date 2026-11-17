class Module:
    def __init__(self):
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        """
        Recursively collect Parameter objects from this module and its submodules.
        """
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
