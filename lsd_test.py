"""
This file is an edited version of the implementation of Will Grathwohl at
https://github.com/wgrathwohl/LSD/blob/master/lsd_test.py
Edited only from line 156 onwards.

- Test: LSD (Learning Stein Discrepancy)
- Paper: [Learning the Stein Discrepancy for Training and Evaluating Energy-Based Models without Sampling](http://proceedings.mlr.press/v119/grathwohl20a/grathwohl20a.pdf)
- Authors: Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Richard Zemel
- Code: [LSD repository](https://github.com/wgrathwohl/LSD) by [Will Grathwohl](https://github.com/wgrathwohl)
"""


import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim
import numpy as np
import networks
import argparse
import os
#import matplotlib
#matplotlib.use('Agg')
import torch.nn.utils.spectral_norm as spectral_norm
from tqdm import tqdm


def try_make_dirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

def randb(size):
    dist = distributions.Bernoulli(probs=(.5 * torch.ones(*size)))
    return dist.sample().float()


class GaussianBernoulliRBM(nn.Module):
    def __init__(self, B, b, c, burn_in=2000):
        super(GaussianBernoulliRBM, self).__init__()
        self.B = nn.Parameter(B)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
        # self.B = B
        # self.b = b
        # self.c = c
        self.dim_x = B.size(0)
        self.dim_h = B.size(1)
        self.burn_in = burn_in

    def score_function(self, x):  # dlogp(x)/dx
        return .5 * torch.tanh(.5 * x @ self.B + self.c) @ self.B.t() + self.b - x

    def forward(self, x):  # logp(x)
        B = self.B
        b = self.b
        c = self.c
        xBc = (0.5 * x @ B) + c
        unden =  (x * b).sum(1) - .5 * (x ** 2).sum(1)# + (xBc.exp() + (-xBc).exp()).log().sum(1)
        unden2 = (x * b).sum(1) - .5 * (x ** 2).sum(1) + torch.tanh(xBc/2.).sum(1)#(xBc.exp() + (-xBc).exp()).log().sum(1)
        #print((unden - unden2).mean())
        assert len(unden) == x.shape[0]
        return unden

    def sample(self, n):
        x = torch.randn((n, self.dim_x)).to(self.B)
        h = (randb((n, self.dim_h)) * 2. - 1.).to(self.B)
        for t in tqdm(range(self.burn_in)):
            x, h = self._blocked_gibbs_next(x, h)
        x, h = self._blocked_gibbs_next(x, h)
        return x

    def _blocked_gibbs_next(self, x, h):
        """
        Sample from the mutual conditional distributions.
        """
        B = self.B
        b = self.b
        # Draw h.
        XB2C = (x @ self.B) + 2.0 * self.c
        # Ph: n x dh matrix
        Ph = torch.sigmoid(XB2C)
        # h: n x dh
        h = (torch.rand_like(h) <= Ph).float() * 2. - 1.
        assert (h.abs() - 1 <= 1e-6).all().item()
        # Draw X.
        # mean: n x dx
        mean = h @ B.t() / 2. + b
        x = torch.randn_like(mean) + mean
        return x, h


class Gaussian(nn.Module):
    def __init__(self, mu, std):
        super(Gaussian, self).__init__()
        self.dist = distributions.Normal(mu, std)

    def sample(self, n):
        return self.dist.sample_n(n)

    def forward(self, x):
        return self.dist.log_prob(x).view(x.size(0), -1).sum(1)


class Laplace(nn.Module):
    def __init__(self, mu, std):
        super(Laplace, self).__init__()
        self.dist = distributions.Laplace(mu, std)

    def sample(self, n):
        return self.dist.sample_n(n)

    def forward(self, x):
        return self.dist.log_prob(x).view(x.size(0), -1).sum(1)


def sample_batch(data, batch_size):
    all_inds = list(range(data.size(0)))
    chosen_inds = np.random.choice(all_inds, batch_size, replace=False)
    chosen_inds = torch.from_numpy(chosen_inds)
    return data[chosen_inds]



def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input,
                               grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)
    return tr_dfdx


def exact_jacobian_trace(fx, x):
    vals = []
    for i in range(x.size(1)):
        fxi = fx[:, i]
        dfxi_dxi = keep_grad(fxi.sum(), x)[:, i][:, None]
        vals.append(dfxi_dxi)
    vals = torch.cat(vals, dim=1)
    return vals.sum(dim=1)


class SpectralLinear(nn.Module):
    def __init__(self, n_in, n_out, max_sigma=1.):
        super(SpectralLinear, self).__init__()
        self.linear = spectral_norm(nn.Linear(n_in, n_out))
        self.scale = nn.Parameter(torch.zeros((1,)))
        self.max_sigma = max_sigma

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.scale) * self.max_sigma


#######################################################################################
#       edited below
#######################################################################################


def generate_rbm_p(
    seed,
    N,
    sigma,
    dx, 
    dh,
    burnin_number=2000,
):
    
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    #print(device)
    
    rs = np.random.RandomState(seed)

    # Model p
    B = rs.randint(0, 2, (dx, dh)) * 2 - 1.0
    b = rs.randn(dx)
    c = rs.randn(dh)

    # Sample from q
    #B_perturbed = B + rs.randn(dx, dh) * sigma

    # convert to torch
    B = torch.from_numpy(B).float().to(device)
    #B_perturbed = torch.from_numpy(B_perturbed)
    b = torch.from_numpy(np.expand_dims(b, 0)).float().to(device)
    c = torch.from_numpy(np.expand_dims(c, 0)).float().to(device)

    p_dist = GaussianBernoulliRBM(B, b, c)

    return p_dist


def lsd_test(data, p_dist, seed):
    
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    #print(device)


    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    N, dim_x = data.shape
    data = torch.from_numpy(data).float().to(device)

    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    n_test = int(N * 0.1)
    weight_decay = 0.0005
    l2 = 0.5
    n_iters = 1000
    lr = 1e-3
    dropout = True
    batch_size = int(N * 0.8)
    maximize_power = True
    log_freq = 10
    val_freq = 100
    val_power = True
    alpha = 0.05
    test = "rbm-pert"
    num_const = 1e-6
    
    data_train = data[:n_train]
    data_rest = data[n_train:]
    data_val = data_rest[:n_val].requires_grad_()
    data_test = data_rest[n_val:].requires_grad_()
    #assert data_test.size(0) = n_test 

    critic = networks.SmallMLP(dim_x, n_out=dim_x, n_hid=300, dropout=dropout).to(device)
    optimizer = optim.Adam(critic.parameters(), lr=lr, weight_decay=weight_decay)

    def stein_discrepency(x, exact=False):
        if "rbm" in test:
            sq = p_dist.score_function(x)
        else:
            logq_u = p_dist(x)
            sq = keep_grad(logq_u.sum(), x)
        fx = critic(x)
        if dim_x == 1:
            fx = fx[:, None]
        sq_fx = (sq * fx).sum(-1)

        if exact:
            tr_dfdx = exact_jacobian_trace(fx, x)
        else:
            tr_dfdx = approx_jacobian_trace(fx, x)

        norms = (fx * fx).sum(1)
        stats = (sq_fx + tr_dfdx)
        return stats, norms

    # training phase
    best_val = -np.inf
    validation_metrics = []
    test_statistics = []
    critic.train()
    for itr in range(n_iters):
        optimizer.zero_grad()
        x = sample_batch(data_train, batch_size)
        x = x.to(device)
        x.requires_grad_()

        stats, norms = stein_discrepency(x)
        mean, std = stats.mean(), stats.std()
        l2_penalty = norms.mean() * l2


        if maximize_power:
            loss = -1. * mean / (std + num_const) + l2_penalty
        elif maximize_adj_mean:
            loss = -1. * mean + std + l2_penalty
        else:
            loss = -1. * mean + l2_penalty

        loss.backward()
        optimizer.step()

        #if itr % log_freq == 0:
            #print("Iter {}, Loss = {}, Mean = {}, STD = {}, L2 {}".format(itr,
            #                                                           loss.item(), mean.item(), std.item(),
            #                                                           l2_penalty.item()))

        if itr % val_freq == 0:
            critic.eval()
            val_stats, _ = stein_discrepency(data_val, exact=True)
            test_stats, _ = stein_discrepency(data_test, exact=True)
            #print("Val: {} +/- {}".format(val_stats.mean().item(), val_stats.std().item()))
            #print("Test: {} +/- {}".format(test_stats.mean().item(), test_stats.std().item()))

            if val_power:
                validation_metric = val_stats.mean() / (val_stats.std() + num_const)
            elif val_adj_mean:
                validation_metric = val_stats.mean() - val_stats.std()
            else:
                validation_metric = val_stats.mean()

            test_statistic = test_stats.mean() / (test_stats.std() + num_const)

            if validation_metric > best_val:
                #print("Iter {}, Validation Metric = {} > {}, Test Statistic = {}, Current Best!".format(itr,
                #                                                                              validation_metric.item(),
                #                                                                              best_val,
                #                                                                              test_statistic.item()))
                best_val = validation_metric.item()
            #else:
                #print("Iter {}, Validation Metric = {}, Test Statistic = {}, Not best {}".format(itr,
                #                                                                            validation_metric.item(),
                #                                                                            test_statistic.item(),
                #                                                                            best_val))
            validation_metrics.append(validation_metric.item())
            test_statistics.append(test_statistic)
            critic.train()
    best_ind = np.argmax(validation_metrics)
    best_test = test_statistics[best_ind]

    #print("Best val is {}, best test is {}".format(best_val, best_test))
    test_stat = best_test * n_test ** .5
    threshold = distributions.Normal(0, 1).icdf(torch.ones((1,)) * (1. - alpha)).item()
    #try_make_dirs(os.path.dirname(args.save))

    if test_stat > threshold:
        return 1
        #print("{} > {}, reject Null".format(test_stat, threshold))
        #f.write("reject")
    else:
        #print("{} <= {}, accept Null".format(test_stat, threshold))
        #f.write("accept")
        return 0
