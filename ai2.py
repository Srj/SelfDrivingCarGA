# v: 2021-05-23T1343 AU
# Issue: "ai.py:51: UserWarning: nn.init.kaiming_normal is now deprecated in favor of nn.init.kaiming_normal_."
# Resolution: renamed `kaiming_normal` to `kaiming_normal_`


import torch.nn as nn
import torch
import random
import numpy as np
import copy
import os
import ai.py


class GA1:
    def __init__(self,
                 pop_mu = 5, pop_lambda = 20):
        self.pop_mu = pop_mu
        self.pop_lambda = pop_lambda

        #Defining mu + lambda population
        self.population = self.pop_mu + self.pop_lambda
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
        scored_models = scored_models[:self.pop_mu]
        self.models = [x[0] for x in scored_models]

        #Save the best one
        if best_model_path is not None:
            unc_model = uncompress_model(self.models[0])
            if os.path.isdir(best_model_path[0]) is False:
                os.makedirs(best_model_path[0])
            torch.save(unc_model, os.path.join(
                best_model_path[0],
                'epoch_' + str(best_model_path[1]) + '.pt'))

        for _ in range(self.pop_mu):
            for _ in range(self.pop_lambda // self.pop_mu):
                print(random.choice(scored_models))
                self.models.append(copy.deepcopy(random.choice(scored_models)[0]))
                self.models[-1].mutate(sigma)
        
        print(len(self.models))

        return median_score, mean_score, max_score


if __name__ == '__main__':
    net = Network()
    print(net)
