import torch
import torch.nn as nn


def get_linear_network_equivalent(
    torch_model: nn.Sequential
):
    # y = W_eq . x + b_eq
    
    W_eq = None
    b_eq = None

    for layer in torch_model:
        if not isinstance(layer, nn.Linear):
            raise ValueError('Model must only contain nn.Linear layers')
        
        W = layer.weight.detach()
        b = layer.bias.detach()

        if W_eq is None:
            W_eq = W
            b_eq = b
        else:
            W_eq = W_eq @ W
            b_eq = W @ b_eq + b

    return W_eq, b_eq