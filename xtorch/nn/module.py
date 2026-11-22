from .parameter import Parameter

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
        seen = set()
        def dfs(node):
            if isinstance(node, Parameter):
                if id(node) not in seen:
                    seen.add(id(node))
                    yield node
            if isinstance(node, Module):
                for value in vars(node).values():
                    yield from dfs(value)
            if isinstance(node, (list, tuple)):
                for value in node:
                    yield from dfs(value)
            if isinstance(node, dict):
                for value in node.values():
                    yield from dfs(value)
        return list(dfs(self))
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
