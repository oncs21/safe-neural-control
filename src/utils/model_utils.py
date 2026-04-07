import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Dense as KDense, Activation as KActivation, Flatten as KFlatten, InputLayer as KInputLayer


# Flatten a torch tensor
class Flatten(nn.Module):
    def forward(self, x):
        y = x.view(x.size(0), -1)
        return y


def _to_torch_tensor(arr):
    try:
        import numpy as _np
        return torch.from_numpy(arr)
    except Exception:
        return torch.tensor(arr.tolist())


# Convert a keras model to a torch model
def keras2torch(
    keras_model
) -> nn.Sequential:
    modules = []

    for l in keras_model.layers:
        if isinstance(l, KInputLayer):
            continue

        if isinstance(l, KDense):
            w, b = l.get_weights()

            # Keras dense layer -> weight matrix shape = (in_features, out_features)
            # Pytorch linear layer -> weight matrix shape = (out_features, in_features)
            in_dim, out_dim = w.shape
            linear = nn.Linear(in_dim, out_dim, bias=True)

            W_t = _to_torch_tensor(w.T).float()
            b_t = _to_torch_tensor(b).float()

            with torch.no_grad():
                linear.weight.copy_(W_t)
                linear.bias.copy_(b_t)

            modules.append(linear)

            act = l.activation
            if act is tf.keras.activations.relu:
                modules.append(nn.ReLU())
            elif act is tf.keras.activations.tanh:
                modules.append(nn.Tanh())
            elif act is tf.keras.activations.sigmoid:
                modules.append(nn.Sigmoid())
            elif act is tf.keras.activations.linear:
                pass
            else:
                raise ValueError(f"Unsupported activation: {act}")

        elif isinstance(l, KActivation):
            act = l.activation
            if act is tf.keras.activations.relu:
                modules.append(nn.ReLU())
            elif act is tf.keras.activations.tanh:
                modules.append(nn.Tanh())
            elif act is tf.keras.activations.sigmoid:
                modules.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation: {act}")

        elif isinstance(l, KFlatten):
            modules.append(Flatten())

        else:
            print(l)
            raise(ValueError("Unknown layer", l))

    torch_model = nn.Sequential(*modules)
    return torch_model