import numpy as np
from .module import Module

class MSELoss(Module):
    def forward(self, prediction, target):
        diff = prediction - target
        return (diff * diff).mean()

class CrossEntropyLoss(Module):
    def forward(self, logits, target):
        shifted = logits - logits.data.max(axis=-1, keepdims=True)
        log_probs = shifted - shifted.exp().sum(axis=-1, keepdims=True).log()
        class_indices = target.data.astype(np.int64)
        one_hot = np.eye(logits.shape[-1])[class_indices]
        
        return -(log_probs * one_hot).sum(axis=-1).mean()