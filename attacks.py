import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from torch.autograd import Variable

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

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
    
    def perturb_inf(self, x_nat, y, rand):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""

        x_nat = x_nat.cpu().numpy()
        y = y.cpu().numpy()

        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape).astype('float32')
            x = np.clip(x, 0, 1) # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        for i in range(self.num_steps):

            x_var = to_var(torch.from_numpy(x), requires_grad=True)
            y_var = to_var(torch.LongTensor(y))

            scores = self.model(x_var)
            loss = F.cross_entropy(scores, y_var)
            loss.backward()
            grad = x_var.grad.data.cpu().numpy()

            x += self.step_size * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 1) # ensure valid pixel range

        return torch.from_numpy(x).cuda()

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

    def perturb_l2(self, x_nat, y, rand):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_2 norm."""
        if self.rand:
            pert = np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            pert_norm = np.linalg.norm(pert)
            pert = pert / max(1, pert_norm)
        else:
            pert = np.zeros(x_nat.shape)

        for i in range(self.num_steps):
            x = x_nat.cpu().numpy() + pert
            x = np.clip(x, 0, 1)
            x = to_var(torch.from_numpy(x).float(), requires_grad=True)

            scores = self.model(x)
            loss = F.cross_entropy(scores, y)
            loss.backward()
            grad = x.grad.data.cpu().numpy()

            normalized_grad = grad / np.linalg.norm(grad)
            pert += self.step_size * normalized_grad

            # project pert to norm ball
            pert_norm = np.linalg.norm(pert)
            rescale_factor = pert_norm / self.epsilon
            pert = pert / max(1, rescale_factor)

            pert = np.clip(np.linalg.norm(pert), -self.epsilon, self.epsilon)

        x = x_nat.cpu().numpy() + pert
        x = np.clip(x, 0, 1)

        return torch.from_numpy(x).float().cuda()

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