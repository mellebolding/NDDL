
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision
import numpy as np
from datetime import date
import os
import pandas as pd
import math
import shutil


import matplotlib.pyplot as plt
import seaborn as sns



def get_stats_named_params(model):
    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone(), 0.0 * param.detach().clone(), 0.0 * param.detach().clone()
        named_params[name] = (param, sm, lm, dm)
    return named_params


def post_optimizer_updates(named_params, alpha=0.2, beta=.5):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        lm.data.add_(-alpha * (param - sm))
        sm.data.mul_((1.0 - beta))
        sm.data.add_(beta * param - (beta / alpha) * lm)


def get_regularizer_named_params(named_params, device, _lambda=1.0, alpha=0.2, rho=0.0):
    regularization = torch.zeros([], device=device)
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho - 1.) * torch.sum(param * lm)
        r_p = _lambda * 0.5 * alpha * torch.sum(torch.square(param - sm))
        regularization += r_p
        # print(name,r_p)
    return regularization


def reset_named_params(named_params):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)


def train_fptt(epoch, batch_size, log_interval,
               train_loader, model, named_params,
               time_steps, k_updates, omega, optimizer,
               clf_alpha, energy_alpha, spike_alpha, clip, lr, device):
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    total_energy_loss = 0
    total_spike_loss = 0
    correct = 0
    model.train()

    # for each batch
    for batch_idx, (data, target) in enumerate(train_loader):

        # to device and reshape
        data, target = data.to(device), target.to(device)
        data = data.view(-1, model.in_dim)

        B = target.size()[0]

        for p in range(time_steps):

            if p == 0:
                h = model.init_hidden(data.size(0))
            elif p % omega == 0:
                h = tuple(v.detach() for v in h)

            o, h = model.forward(data, h)
            # wandb.log({
            #         'rec layer adap threshold': h[5].detach().cpu().numpy(),
            #         'rec layer mem potential': h[3].detach().cpu().numpy()
            #     })

            # get prediction
            if p == (time_steps - 1):
                pred = o.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            if p % omega == 0 and p > 0:
                optimizer.zero_grad()

                # classification loss
                clf_loss = (p + 1) / k_updates * F.nll_loss(o, target)
                # clf_loss = snr*F.cross_entropy(output, target,reduction='none')
                # clf_loss = torch.mean(clf_loss)

                # regularizer loss
                regularizer = get_regularizer_named_params(named_params, device, _lambda=1.0)

                # mem potential loss take l1 norm / num of neurons /batch size
                if len(model.hidden_dims) == 2:
                    energy = (torch.sum(model.error1 ** 2) + torch.sum(model.error2 ** 2)) / B / sum(model.hidden_dims)
                    spike_loss = (torch.sum(h[1]) + torch.sum(h[5])) / B / sum(model.hidden_dims)
                elif len(model.hidden_dims) == 3:
                    # energy = (torch.sum(model.error1 ** 2) + torch.sum(model.error2 ** 2) + torch.sum(model.error3 ** 2)) / B / sum(model.hidden_dims)
                    energy = (torch.sum(torch.abs(model.error1)) + torch.sum(torch.abs(model.error2)) + torch.sum(torch.abs(model.error3))) / B / sum(model.hidden_dims)
                    spike_loss = (torch.sum(h[1]) + torch.sum(h[5]) + torch.sum(h[9])) / B / sum(model.hidden_dims)


                # overall loss
                loss = clf_alpha * clf_loss + regularizer + energy_alpha * energy + spike_alpha * spike_loss

                loss.backward()

                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                optimizer.step()
                post_optimizer_updates(named_params)

                train_loss += loss.item()
                total_clf_loss += clf_loss.item()
                total_regularizaton_loss += regularizer  # .item()
                total_energy_loss += energy.item()
                total_spike_loss += spike_loss.item()


                model.error1 = 0
                model.error2 = 0
                if len(model.hidden_dims) == 3:
                    model.error3 = 0


        if batch_idx > 0 and batch_idx % log_interval == (log_interval - 1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tenerg: {:.6f}\tlr: {:.6f}\ttrain acc:{:.4f}\tLoss: {:.6f}\
                \tClf: {:.6f}\tReg: {:.6f}\tFr_p: {:.6f}\tFr_r: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_energy_loss / log_interval,
                      lr, 100 * correct / (log_interval * B),
                       train_loss / log_interval,
                       total_clf_loss / log_interval, total_regularizaton_loss / log_interval,
                       model.fr_layer2 / time_steps / log_interval,
                       model.fr_layer1 / time_steps / log_interval))


            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0
            total_energy_loss = 0
            total_spike_loss = 0
            correct = 0
            # model.network.fr = 0
            model.fr_layer2 = 0
            model.fr_layer1 = 0
            if len(model.hidden_dims) == 3:
                model.fr_layer3 = 0


