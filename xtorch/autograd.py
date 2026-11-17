import numpy as np


def topological_sort(tensor):
    """
    Return all nodes in topological order with parents appearing before their children.
    """
    raise NotImplementedError


def unbroadcast(grad, shape):
    """
    Reduce a broadcasted gradient back to the original operand shape.
    """
    raise NotImplementedError
