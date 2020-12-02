import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from torch.autograd import Variable

class PGDAttack:

    def __init__(self, model, epsilon, num_steps, step_size, rand):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = rand

    def perturb_inf_v2(self, x, y):
        x_nat = x.clone().detach()

        if self.rand:
            new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.epsilon
            x = torch.clamp(new_x, 0, 1)

        else:
            x = x + torch.zeros_like(x)

        for i in range(self.num_steps):
            x = x.clone().detach().requires_grad_(True)
            scores = self.model(x)
            loss = F.cross_entropy(scores, y)
            loss.backward()
            grad = x.grad.data

            x = x.clone().detach() + torch.sign(grad) * self.step_size

            diff = x - x_nat
            diff = diff = torch.clamp(diff, -self.epsilon, self.epsilon)
            x = torch.clamp(x_nat + diff, 0, 1)

        return x

    def perturb_l2_v2(self, x, y):

        x_nat = x.clone().detach()

        if self.rand:
            l = len(x.shape) - 1
            rp = torch.randn_like(x)
            rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
            x = torch.clamp(x + self.epsilon * rp / (rp_norm + 1e-10), 0, 1)
        else:
            x = x + torch.zeros_like(x)
    
        for i in range(self.num_steps):
            x = x.clone().detach().requires_grad_(True)
            scores = self.model(x)
            loss = F.cross_entropy(scores, y)
            loss.backward()
            grad = x.grad.data

            l = len(x.shape) - 1
            g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
            scaled_g = grad / (g_norm + 1e-10)
            x = x.clone().detach() + scaled_g * self.step_size

            diff = x - x_nat
            diff = diff.renorm(p=2, dim=0, maxnorm=self.epsilon)
            x = torch.clamp(x_nat + diff, 0, 1)

        return x