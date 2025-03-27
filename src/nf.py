import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class _Bijection(nn.Module):
    def __init__(self):
        super(_Bijection, self).__init__()

    def forward(self, x):
        pass

    def inverse(self, z):
        pass



class _ActNormBijection(_Bijection):

    def __init__(self, num_features, data_dep_init=True, eps=1e-6):
        super(_ActNormBijection, self).__init__()
        self.num_features = num_features
        self.data_dep_init = data_dep_init
        self.eps = eps

        self.register_buffer('initialized', torch.zeros(1) if data_dep_init else torch.ones(1))
        self.register_params()

    def data_init(self, x):
        self.initialized += 1.
        with torch.no_grad():
            x_mean, x_std = self.compute_stats(x)
            self.shift.data = x_mean
            self.log_scale.data = torch.log(x_std + self.eps)

    def forward(self, x):
        if self.training and not self.initialized: self.data_init(x)
        z = (x - self.shift) * torch.exp(-self.log_scale)
        ldj = torch.sum(-self.log_scale).expand([x.shape[0]]) * self.ldj_multiplier(x)
        return z, ldj

    def inverse(self, z):
        return self.shift + z * torch.exp(self.log_scale)

    def register_params(self):
        '''Register parameters shift and log_scale'''
        raise NotImplementedError()

    def compute_stats(self, x):
        '''Compute x_mean and x_std'''
        raise NotImplementedError()

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        raise NotImplementedError()


class ActNorm(_ActNormBijection):
    
    def register_params(self):
        '''Register parameters shift and log_scale'''
        self.register_parameter('shift', nn.Parameter(torch.zeros(1, self.num_features)))
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, self.num_features)))

    def compute_stats(self, x):
        '''Compute x_mean and x_std'''
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True)
        return x_mean, x_std

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        return 1



class Conv1x1(_Bijection):

    def __init__(self, input_dim):
        super(Conv1x1, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.orthogonal_(self.weight)
        self._inverse = torch.inverse(self.weight)

    def forward(self, x):
        z = F.conv1d(x.unsqueeze(-1), self.weight.unsqueeze(-1)).squeeze(-1)
        ldj = (torch.slogdet(self.weight)[1]).expand([x.shape[0]])
        # self._inverse = torch.inverse(self.weight)
        return z, ldj
    
    def inverse(self, z):
        return F.conv1d(z.unsqueeze(-1), self._inverse.unsqueeze(-1)).squeeze(-1)


class AffineCouplingLayer(_Bijection):
    def __init__(self, net):
        super(AffineCouplingLayer, self).__init__()
        self.net = net
        self.m = net.dim

    def forward(self, x): # NxD
        x1, x2 = torch.chunk(x, dim=1, chunks=2)
        log_s, t = self.net(x1)
        z1 = x1
        z2 = torch.exp(log_s) * x2 + t
        z = torch.cat((z1, z2), 1)
        log_det = log_s.sum(1)
        return z, log_det # NxD , N

    def inverse(self, y): # NxD
        y1, y2 = torch.chunk(y, dim=1, chunks=2)
        x1 = y1
        log_s, t = self.net(x1)
        x2 = (y2 - t) / torch.exp(log_s)
        x = torch.cat((x1, x2), 1)
        return x # NxD

class SwitchSides(_Bijection):

    def forward(self, x):
        x1, x2 = torch.chunk(x, dim=1, chunks=2)
        y = torch.cat((x2, x1), 1)
        return y, 0.

    def inverse(self, z):
        x1, x2 = torch.chunk(z, dim=1, chunks=2)
        x = torch.cat((x2, x1), 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.dim = dim

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU(inplace=True)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += identity
        out = self.relu(out)

        # log_s, t = torch.chunk(out, dim=1, chunks=2)

        return out
    

class ResNet(nn.Module):
    def __init__(self, in_dim, num_blocks=1):
        super(ResNet, self).__init__()
        self.dim=in_dim
        out_dim = 2*in_dim
        mid_dim= 2*in_dim
        layers = [nn.Linear(in_dim, mid_dim)] +\
                 [ResidualBlock(mid_dim) for _ in range(num_blocks)] +\
                 [nn.Linear(mid_dim, out_dim)]

        nn.init.zeros_(layers[-1].weight)
        if hasattr(layers[-1], 'bias'):
            nn.init.zeros_(layers[-1].bias)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        log_s, t = torch.chunk(out, dim=1, chunks=2)
        return log_s, t


class SimpleTransform(nn.Module):
    def __init__(self, dim, inflate_coef=2):
        super(SimpleTransform, self).__init__()
        self.dim = dim
        internal_dim = int(dim * inflate_coef)
        self.model = nn.Sequential(
            nn.Linear(dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, 2 * dim),
        )
        nn.init.zeros_(self.model[-1].weight)
        nn.init.zeros_(self.model[-1].bias)


    def forward(self, x):
        out = self.model(x)
        log_s, t = torch.chunk(out, dim=1, chunks=2)
        return log_s, t



class NormalizingFlow(nn.Module):
    def __init__(self, input_dim, num_steps=2):
        super(NormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.transforms = []
        for i in range(num_steps):
            self.transforms.append(AffineCouplingLayer(SimpleTransform(input_dim//2, 0.5)))
            if i != num_steps-1:
                self.transforms.append(SwitchSides())
        self.transforms = nn.Sequential(*self.transforms)

        self.register_buffer('loc', torch.zeros(input_dim))
        self.register_buffer('log_scale', torch.zeros(input_dim))

    def z_dist(self):
        z_dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale), validate_args=False)
        return z_dist

    def log_prob(self, x, norm=True):
        z = x
        log_abs_det = 0.
        for m in self.transforms:
            z, ld_layer = m(z)
            log_abs_det += ld_layer
        log_pz = self.z_dist().log_prob(z).sum(-1)
        log_px = (log_pz + log_abs_det) / self.input_dim
        return log_px

    def forward(self, x):
        z = x
        for m in self.transforms:
            z, _ = m(z)
        return z

    def inverse(self, z):
        for m in reversed(self.transforms):
            z = m.inverse(z)
        return z

    def sample(self, num_samples, T=1):
        z_dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale))
        z = z_dist.sample(torch.Size([num_samples])) * T
        x = self.inverse(z)
        return x
    


class MiniGlow(nn.Module):
    def __init__(self, input_dim=256, num_steps=2):
        super(MiniGlow, self).__init__()
        self.input_dim = input_dim

        self.transforms = []
        for i in range(num_steps):
            self.transforms.append(ActNorm(input_dim))
            self.transforms.append(Conv1x1(input_dim))
            self.transforms.append(AffineCouplingLayer(SimpleTransform(input_dim//2, 2)))
            # if i != num_steps-1:
            #     self.transforms.append(SwitchSides())

        self.transforms = nn.Sequential(*self.transforms)

        self.register_buffer('loc', torch.zeros(input_dim))
        self.register_buffer('log_scale', torch.zeros(input_dim))

    def z_dist(self):
        z_dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale))
        return z_dist

    def log_prob(self, x):
        z = x
        log_abs_det = 0.
        for m in self.transforms[1:]:
            z, ld_layer = m(z)
            log_abs_det += ld_layer
        log_pz = self.z_dist().log_prob(z).sum(-1)
        log_px = (log_pz + log_abs_det) / self.input_dim
        return log_px

    def forward(self, x):
        z = x
        for m in self.transforms:
            z, _ = m(z)
        return z

    def inverse(self, z):
        for m in reversed(self.transforms):
            z = m.inverse(z)
        return z

    def sample(self, num_samples, T=1):
        z_dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale))
        z = z_dist.sample(torch.Size([num_samples])) * T
        x = self.inverse(z)
        return x



if __name__ == '__main__':

    # nf = NormalizingFlow(304, num_steps=2)
    nf = MiniGlow(256, num_steps=2)

    prelogits = torch.randn(1000, 256)
    # z = nf(prelogits)
    ln_px = nf.log_prob(prelogits)
    print(- ln_px)
    print(nf.sample(2).shape)

    x = torch.randn(1000, 256) * 100
    z = nf.forward(x)
    x_ = nf.inverse(z)
    print((x - x_).abs().max())
    

