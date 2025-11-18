import torch.nn
from torch import nn as nn
import numpy as np


class ResidLinear(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(nin, nout, bias=bias)

    def forward(self, x):
        z = self.linear(x) + x
        return z


class MLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation, bias=True, norm=None):
        super(MLP, self).__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim, bias=bias),
            activation(inplace=True),
        ]
        for _ in range(nlayers):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(activation(inplace=True))
        if norm is not None:
            if norm == "instance":
                layers.append(nn.InstanceNorm1d(hidden_dim))
            elif norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation, bias=True, norm=False):
        super(ResidLinearMLP, self).__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim, bias = bias),
            activation(inplace=True),
        ]
        for _ in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim, bias = bias))
            layers.append(activation(inplace=True))

        if norm:
            layers.append(nn.InstanceNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim, bias = bias))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)



class HetGMMVAESimpleIndependent(nn.Module):
    def __init__(
        self,
        qlayers,
        qdim,
        in_dim,
        zdim,
        gaussian_embedding_dim=128,
        gaussian_kdim=256,
        gaussian_klayers=3,
        feature_dim=256,
        feature_kdim=256,
        feature_klayers=3,
        activation=nn.ReLU,
        archi_type='MLP'
    ):
        super(HetGMMVAESimpleIndependent, self).__init__()
        self.zdim = zdim
        self.in_dim = in_dim
        if archi_type == 'MLP':
            self.encoder = MLP(
                in_dim,
                qlayers,
                qdim,
                zdim * 2,
                activation,
                bias=True,
                norm="batch"
            )
        elif archi_type == 'Resid':
            self.encoder = ResidLinearMLP(
                in_dim,
                qlayers,
                qdim,
                zdim * 2,
                activation,
            )
        else:
            raise RuntimeError("Encoder mode {} not recognized".format(archi_type))

        self.gaussian_embedding_dim = gaussian_embedding_dim
        gaussian_info_dim = 5
        self.gaussian_embedding = self.make_decoder(
            gaussian_info_dim,
            gaussian_kdim,
            gaussian_klayers,
            self.gaussian_embedding_dim,
            archi_type,
            activation=activation,
            bias=True,
            norm="layer",
            # norm=None,
        )

        self.feature_dim = feature_dim
        feature_info_dim = zdim + self.gaussian_embedding_dim
        self.feature_decoder = self.make_decoder(
            feature_info_dim,
            feature_kdim,
            feature_klayers,
            self.feature_dim,
            archi_type,
            activation=activation,
            bias=True,
            # norm="layer",
            norm=None,
        )
        self.d_density = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim, bias=True),
            activation(inplace=True),
            nn.Linear(self.feature_dim, 1, bias=True),
        )
        self.d_scaling = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim, bias=True),
            activation(inplace=True),
            nn.Linear(self.feature_dim, 1, bias=True),
        )
        self.d_xyz = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim, bias=True),
            activation(inplace=True),
            nn.Linear(self.feature_dim, 3, bias=True),
        )


    def make_decoder(self, in_dim, hidden_dim, nlayers, out_dim, archi_type, activation=nn.ReLU, bias=True, norm=None):
        layers = [
            nn.Linear(in_dim, hidden_dim, bias = bias),
            activation(inplace=True),
        ]
        for _ in range(nlayers):
            if archi_type == "MLP":
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            else:
                layers.append(ResidLinear(hidden_dim, hidden_dim, bias=bias))
            layers.append(activation(inplace=True))

        layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))
        # layers.append(activation(inplace=True))
        if norm is not None:
            # layers.append(activation(inplace=True))
            if norm == "instance":
                layers.append(nn.InstanceNorm1d(out_dim))
            elif norm == "layer":
                layers.append(nn.LayerNorm(out_dim))
        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, img):
        img = img.view(img.shape[0], -1)  # flatten input
        z = self.encoder(img)
        return z[:, :self.zdim], z[:, self.zdim:]

    def decode(self, z, gaussian_info):
        B = z.shape[0]
        z = z.unsqueeze(1).expand(-1, gaussian_info.shape[0], -1)
        gaussian_info = gaussian_info.view(1, *gaussian_info.shape).expand(B, -1, -1)
        gaussian_embedding = self.gaussian_embedding(gaussian_info)
        fea_input = torch.cat([z, gaussian_embedding], dim=2)
        fea_output = self.feature_decoder(fea_input)
        d_density = self.d_density(fea_output)
        d_scaling = self.d_scaling(fea_output)
        d_xyz = self.d_xyz(fea_output)
        return d_density, d_scaling, d_xyz, gaussian_embedding
    
    def forward(self, img, gaussian_density, gaussian_scaling, gaussian_pos, encode_only=False):
        z_mu, z_logvar = self.encode(img)
        z = self.reparameterize(z_mu, z_logvar)
        if encode_only:
            return z_mu, z_logvar
        else:
            gaussian_info = torch.cat([gaussian_density, gaussian_scaling, gaussian_pos], dim=-1)
            return self.decode(z, gaussian_info), z_mu, z_logvar

