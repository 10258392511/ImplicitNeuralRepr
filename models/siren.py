# Credited to Sébastien Emery at ETH (https://ee.ethz.ch/the-department/people-a-z/person-detail.MzA1Mzk1.TGlzdC8zMjc5LC0xNjUwNTg5ODIw.html) for SirenLayer and Siren

import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


class SineLayer(nn.Module):
    """
    Taken from: https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
    """

    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class SirenComplex(nn.Module):
    def __init__(self, params):
        """
        params: in_features, hidden_features, hidden_layers, out_features, 
                outermost_linear: bool, first_omega_0, hidden_omega_0
        """
        super().__init__()
        self.params = params
        self.siren_real = Siren(**self.params)
        self.siren_imag = Siren(**self.params)
    
    def forward(self, x: torch.Tensor):
        x = x
        out_real = self.siren_real(x)
        out_imag = self.siren_imag(x)
        x_out = out_real + 1j * out_imag

        return x_out
    