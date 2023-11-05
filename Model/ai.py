# v: 2021-05-23T1343 AU
# Issue: "GA2.py:51: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_."
# Resolution: renamed `kaiming_normal` to `kaiming_normal_`


import torch.nn as nn
import torch
import random
import numpy as np
import copy
import os


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(5, 20)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(20, 2)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x


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


def random_state():
    return random.randint(0, 2 ** 31 - 1)


class CompressedModel:
    def __init__(self, start_rng=None, other_rng=None):
        self.start_rng = start_rng if start_rng is not None else random_state()
        self.other_rng = other_rng if other_rng is not None else []

    def mutate(self, sigma, rng_state=None):
        self.other_rng.append(
            (sigma, rng_state if rng_state is not None else random_state())
        )


# class GA:
#     def __init__(self,
#                  pop_mu = 5, pop_lambda = 20):
#         self.pop_mu = pop_mu
#         self.pop_lambda = pop_lambda

#         #Defining mu + lambda population
#         self.population = self.pop_mu + self.pop_lambda
#         self.models = [CompressedModel() for _ in range(self.population)]

#     def get_best_models(self, cars):
#         scores = [car.reward() for car in cars]
#         scored_models = list(zip(self.models, scores))
#         scored_models.sort(key=lambda x: x[1], reverse=True)
#         return scored_models

#     def evolve_iter(self, cars, sigma=0.05,
#                     best_model_path=None):
#         scored_models = self.get_best_models(cars)
#         scores = [s for _, s in scored_models]
#         median_score = np.median(scores)
#         mean_score = np.mean(scores)
#         max_score = scored_models[0][1]

#         # Method 1
#         # choose top mu models
#         scored_models = scored_models[:self.pop_mu]
#         self.models = [x[0] for x in scored_models]

#         #Save the best one
#         if best_model_path is not None:
#             unc_model = uncompress_model(self.models[0])
#             if os.path.isdir(best_model_path[0]) is False:
#                 os.makedirs(best_model_path[0])
#             torch.save(unc_model, os.path.join(
#                 best_model_path[0],
#                 'epoch_' + str(best_model_path[1]) + '.pt'))

#         for _ in range(self.pop_mu):
#             for _ in range(self.pop_lambda // self.pop_mu):
#                 print(random.choice(scored_models))
#                 self.models.append(copy.deepcopy(random.choice(scored_models)[0]))
#                 self.models[-1].mutate(sigma)
        
#         print(len(self.models))

#         return median_score, mean_score, max_score


def crossover(model1, model2):
    #Implementing Uniform Crossover
    model1 = uncompress_model(model1)
    model2 = uncompress_model(model2)
    model1_params = model1.state_dict()
    model2_params = model2.state_dict()
    child_params = {}
    for name, param in model1_params.items():
        child_params[name] = param if random.random() < 0.5 else model2_params[name]


class GA:
    def __init__(self, pop_mu = 5, population  = 25):
        self.pop_mu = pop_mu
        self.population = population 
        self.models = [CompressedModel() for _ in range(self.population)]

    def get_best_models(self, cars):
        scores = [car.reward() for car in cars]
        scored_models = list(zip(self.models, scores))
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models

    def evolve_iter(self, cars, sigma=0.05,
                    best_model_path=None):
        scored_models = self.get_best_models(cars)
        scores = [s for _, s in scored_models]
        median_score = np.median(scores)
        mean_score = np.mean(scores)
        max_score = scored_models[0][1]

        # Method 1
        # choose top mu models
        self.models = [copy.deepcopy(x[0]) for x in scored_models[:self.pop_mu]]

        #Save the best one
        if best_model_path is not None:
            unc_model = uncompress_model(self.models[0])
            if os.path.isdir(best_model_path[0]) is False:
                os.makedirs(best_model_path[0])
            torch.save(unc_model, os.path.join(
                best_model_path[0],
                'epoch_' + str(best_model_path[1]) + '.pt'))

        

        for _ in range((self.population - self.pop_mu) // 2):

            #choose parent for crossover
            model1 = random.choice(scored_models)[0]
            model2 = random.choice(scored_models)[0]
            c1, c2 = crossover(model1, model2)

            self.models.append(copy.deepcopy(random.choice(scored_models)[0]))
            self.models[-1].mutate(sigma)
        
        return median_score, mean_score, max_score


if __name__ == '__main__':
    net = Network()
    print(net)
