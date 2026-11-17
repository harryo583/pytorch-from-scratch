import numpy as np


def topological_sort(tensor):
    """
    Return all nodes in topological order with parents appearing before their children.
    """
    topological_order = []
    visited = set()
    
    def build_topological_order(node):
        if node not in visited:
            visited.add(node)
            for parent in node._prev:
                build_topological_order(parent)
            topological_order.append(node)
    
    build_topological_order(tensor)

    return topological_order


def unbroadcast(grad, shape):
    """
    Reduce a broadcasted gradient back to the original operand shape.
    """
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
        
    return grad