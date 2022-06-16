

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten(t):
    return [item for sublist in t for item in sublist]


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, activation_fn=F.leaky_relu):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.activation_fn = activation_fn
        neurons = [input_dim, *hidden_dim, latent_dim]
        if cond_dropout:
            layers = flatten(
                [[nn.Linear(neurons[i - 1], neurons[i]), nn.Dropout(p=drop_rate)] for i in range(1, len(neurons))])
            self.hidden = nn.ModuleList(layers[:-1])
        else:
            linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)

    def forward(self, x):
        for layer in self.hidden:
            x = self.activation_fn(layer(x))
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, actFn, activation_fn=F.leaky_relu):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.actFn = actFn
        self.activation_fn = activation_fn
        neurons = [latent_dim, *reversed(hidden_dim)]
        if cond_dropout:
            layers = flatten(
                [[nn.Linear(neurons[i-1], neurons[i]), nn.Dropout(p=drop_rate)] for i in range(1, len(neurons))])
            self.hidden = nn.ModuleList(layers)
        else:
            linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)
        self.fc = nn.Linear(neurons[-1], input_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = self.activation_fn(layer(x))
        if self.actFn == 'sigmoid':
            x = torch.sigmoid(self.fc(x))  # last layer before output is sigmoid, transfer between 0 and 1
        elif self.actFn == 'tanh':
            x = torch.tanh(self.fc(x))
        elif self.actFn == 'leakyRelu':
            x = F.leaky_relu(self.fc(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, actFn):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.actFn = actFn
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.latent_dim, self.cond_dropout, self.drop_rate)
        self.decoder = Decoder(self.input_dim, self.hidden_dim, self.latent_dim, self.cond_dropout, self.drop_rate, self.actFn)

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon, latent


class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, activation_fn=F.leaky_relu):
        super(VAE_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.activation_fn = activation_fn
        neurons = [input_dim, *hidden_dim]
        if cond_dropout:
            layers = flatten(
                [[nn.Linear(neurons[i - 1], neurons[i]), nn.Dropout(p=drop_rate)] for i in range(1, len(neurons))])
            self.hidden = nn.ModuleList(layers[:-1])
        else:
            linear_layers = [nn.Linear(neurons[i - 1], neurons[i]) for i in range(1, len(neurons))]
            self.hidden = nn.ModuleList(linear_layers)
        self.fc_mu = nn.Linear(in_features=hidden_dim[-1], out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_dim[-1], out_features=latent_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = self.activation_fn(layer(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, actFn):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.actFn = actFn
        self.encoder = VAE_Encoder(self.input_dim, self.hidden_dim, self.latent_dim, self.cond_dropout, self.drop_rate)
        self.decoder = Decoder(self.input_dim, self.hidden_dim, self.latent_dim, self.cond_dropout, self.drop_rate, self.actFn)

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu



class SAEIBS(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dropout, drop_rate, actFn):
        super(SAEIBS, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dropout = cond_dropout
        self.drop_rate = drop_rate
        self.actFn = actFn
        self.encoder = Encoder(self.input_dim, self.hidden_dim[:-1], self.hidden_dim[-1], self.cond_dropout, self.drop_rate)
        self.decoder = Decoder(self.input_dim, self.hidden_dim[:-1], self.hidden_dim[-1], self.cond_dropout, self.drop_rate, self.actFn)
        # svd related
        self.V = nn.Parameter(torch.empty(self.latent_dim, self.latent_dim))
        self.rank = self.latent_dim
        self.emb = None
        self.mean_emb = nn.Parameter(torch.empty(self.latent_dim,), requires_grad=False)

    def initialize_svd(self, x, ibs_all):
        self.emb = torch.mm(ibs_all, x)
        self.mean_emb = nn.Parameter(torch.mean(self.emb, 0))
        _, _, V = torch.svd_lowrank(self.emb - self.mean_emb, self.rank)
        max_ind = torch.argmax(torch.abs(V), 0)
        colsign = torch.sign(V[max_ind, torch.arange(V.shape[1])])
        self.V = nn.Parameter(V * colsign)

    def update_svd(self, x, ibs_batch, indices=None):
        with torch.no_grad():
            self.emb[indices.squeeze(), :] = torch.mm(ibs_batch, x)
            self.mean_emb = nn.Parameter(torch.mean(self.emb, 0))
        _, _, V = torch.svd_lowrank(self.emb - self.mean_emb, self.rank)
        max_ind = torch.argmax(torch.abs(V), 0)
        colsign = torch.sign(V[max_ind, torch.arange(V.shape[1])])
        self.V = nn.Parameter(V * colsign)
        z = torch.matmul(self.emb - self.mean_emb, self.V)
        x_hat = torch.matmul(z, torch.transpose(self.V, 0, 1)) + self.mean_emb
        return z, x_hat, self.V, self.mean_emb

    def encoder_svd(self, x, ibs_batch, indices=None):
        x_enc = self.encoder(x)
        z, x_hat, self.V, self.mean_emb = self.update_svd(x_enc, ibs_batch, indices)
        return x_enc, z, x_hat, self.V, self.mean_emb

    def forward(self, x, ibs_batch, indices=None):
        x_enc, z, x_hat, self.V, self.mean_emb = self.encoder_svd(x, ibs_batch, indices)
        x_back = torch.mm(torch.inverse(ibs_batch), x_hat[indices.squeeze(), :])
        x_recon = self.decoder(x_back)
        # print(pdist(x_enc, x_back).mean())
        return x_recon, z, self.V, self.mean_emb


