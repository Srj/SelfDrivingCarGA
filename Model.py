import random

import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, rng_state):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(5, 20)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(20, 2)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()

        self.rng_state = rng_state
        torch.manual_seed(rng_state)

        self.evolve_states = []

        self.add_tensors = {}
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal_(tensor)
            else:
                tensor.data.zero_()

    def forward(self, x):
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x

    def evolve(self, sigma, rng_state):
        torch.manual_seed(rng_state)
        self.evolve_states.append((sigma, rng_state))

        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)

    def compress(self):
        return CompressedModel(self.rng_state, self.evolve_states)


def uncompress_model(model):
    start_rng, other_rng = model.start_rng, model.other_rng
    m = Model(start_rng)
    for sigma, rng in other_rng:
        m.evolve(sigma, rng)
    return m


class CompressedModel:
    def __init__(self, start_rng=None, other_rng=None):
        self.start_rng = start_rng if start_rng is not None else random_state()
        self.other_rng = other_rng if other_rng is not None else []

    def mutate(self, sigma, rng_state=None):
        self.other_rng.append(
            (sigma, rng_state if rng_state is not None else random_state())
        )


def random_state():
    return random.randint(0, 2 ** 31 - 1)
