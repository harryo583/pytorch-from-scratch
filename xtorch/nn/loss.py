from .module import Module


class MSELoss(Module):
    def forward(self, prediction, target):
        diff = prediction - target
        return (diff * diff).mean()

class CrossEntropyLoss(Module):
    def forward(self, logits, target):
        raise NotImplementedError
