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

from .utils import gaussian, ActFun_adp

# layers
# the strength at which the voltage from the apical tuft (Va,i(t)) drives the soma
def shifted_sigmoid(currents):
    return (1 / (1 + torch.exp(-currents)) - 0.5)/2


class SnnLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            is_rec: bool,
            is_adapt: bool,
            one_to_one: bool,
            act_fun_adp: ActFun_adp, #XXX
            device, #XXX
            tau_m_init=15.,
            tau_adap_init=20,
            tau_a_init=15.,
            dt = 0.5,
            bias = True
    ):
        super(SnnLayer, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.is_rec = is_rec
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one
        self.dt = dt
        self.device = device #XXX
        self.act_fun_adp = act_fun_adp #XXX

        # print('in_dim', self.in_dim)
        # print('hidden_dim', self.hidden_dim)
        # num_excitatory = int(0.75 * in_dim)  # 75% excitatory neurons
        # num_inhibitory = self.in_dim - num_excitatory

        # self.neuron_mask = torch.cat([
        #     torch.ones(num_excitatory, device=device),  # Excitatory neurons
        #     torch.zeros(num_inhibitory, device=device)  # Inhibitory neurons
        # ]).bool()
        # self.neuron_mask = self.neuron_mask[torch.randperm(self.neuron_mask.size(0))]  # Shuffle randomly
        # # # TODO correct shape
        # self.neuron_mask = self.neuron_mask.unsqueeze(1).repeat(1, hidden_dim).t()

        # print('neuron mask', self.neuron_mask)
        
        # kijken hoe de verdeling is en mask daarop aanpassen
        # gewichtsmatrix diagonaal op 1 houden
        if is_rec:
            self.rec_w = DaleLinearLayer(hidden_dim, hidden_dim, bias=bias)
            # init weights
            if bias:
                nn.init.constant_(self.rec_w.bias, 0)
            nn.init.xavier_uniform_(self.rec_w.weight)

            p = torch.full(self.rec_w.weight.size(), fill_value=0.5).to(device)
            self.weight_mask = torch.bernoulli(p)

        else:
            self.fc_weights = DaleLinearLayer(in_dim, hidden_dim, bias=bias)
            if bias:
                nn.init.constant_(self.fc_weights.bias, 0)
            nn.init.xavier_uniform_(self.fc_weights.weight)

        print('fc weights', self.fc_weights.weight.shape)

        # define param for time constants
        self.tau_adp = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_m = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_a = nn.Parameter(torch.Tensor(hidden_dim))

        nn.init.normal_(self.tau_adp, tau_adap_init, .1)
        nn.init.normal_(self.tau_m, tau_m_init, .1)
        nn.init.normal_(self.tau_a, tau_a_init, 0.1)

        # self.tau_adp = nn.Parameter(torch.Tensor(1))
        # self.tau_m = nn.Parameter(torch.Tensor(1))
        # self.tau_a = nn.Parameter(torch.Tensor(1))

        # nn.init.constant_(self.tau_adp, tau_adap_init)
        # nn.init.constant_(self.tau_m, tau_m_init)
        # nn.init.constant_(self.tau_a, tau_a_init)

        # nn.init.normal_(self.tau_adp, 200., 20.)
        # nn.init.normal_(self.tau_m, 20., .5)

        self.sigmoid = nn.Sigmoid()

    def mem_update(self, ff, fb, soma, spike, a_curr, b, is_adapt, baseline_thre=0.1, r_m=3):
        """
        mem update for each layer of neurons
        :param ff: feedforward signal
        :param fb: feedback signal to apical tuft
        :param soma: mem voltage potential at soma
        :param spike: spiking at last time step
        :param a_curr: apical tuft current at last t
        :param b: adaptive threshold
        :param baseline_thre: baseline threshold is b_j0 XXX
        :return:
        """
        # alpha = self.sigmoid(self.tau_m)
        # rho = self.sigmoid(self.tau_adp)
        # eta = self.sigmoid(self.tau_a)
        alpha = torch.exp(-self.dt/self.tau_m)
        rho = torch.exp(-self.dt/self.tau_adp)
        eta = torch.exp(-self.dt/self.tau_a)

        if is_adapt:
            beta = 1.8
        else:
            beta = 0.

        b = rho * b + (1 - rho) * spike  # adaptive contribution
        new_thre = baseline_thre + beta * b  # udpated threshold

        current_new = ff 


        a_new = eta * a_curr + fb  # fb into apical tuft

        soma_new = alpha * soma + shifted_sigmoid(a_new) + current_new - new_thre * spike
        # soma_new = alpha * soma + 1/2 * (a_new) + ffs - new_thre * spike

        inputs_ = soma_new - new_thre

        spike = self.act_fun_adp(inputs_)  # act_fun : approximation firing function
        # mem = (1 - spike) * mem

        return soma_new, spike, a_new, new_thre, b

    def forward(self, ff, fb, soma_t, spk_t, a_curr_t, b_t):
        """
        forward function of a single layer. given previous neuron states and current input, update neuron states

        :param ff: ff signal (not counting rec)
        :param fb: fb top down signal
        :param soma_t: soma voltage
        :param a_curr_t: apical tuft voltage
        :return:
        """

        if self.is_rec:
            self.rec_w.weight.data = self.rec_w.weight.data * self.weight_mask
            # self.rec_w.weight.data = (self.rec_w.weight.data < 0).float() * self.rec_w.weight.data
            r_in = ff + self.rec_w(spk_t)
        else:
            if self.one_to_one:
                r_in = ff
            else:
                r_in = self.fc_weights(ff)

        soma_t1, spk_t1, a_curr_t1, _, b_t1 = self.mem_update(r_in, fb, soma_t, spk_t, a_curr_t, b_t, self.is_adapt)

        return soma_t1, spk_t1, a_curr_t1, b_t1
    

    def clip_weights(self):
        """Clip weights based on excitatory/inhibitory mask."""

        # Apply masks: excitatory neurons (positive weights), inhibitory neurons (negative weights)
        # excitatory_mask = self.neuron_mask.bool()  # Excitatory mask
        # inhibitory_mask = (1 - self.neuron_mask).bool()  # Inhibitory mask

        # if self.is_rec:
        #     excitatory_negative_mask = (self.rec_w.weight.data < 0) & excitatory_mask
        #     inhibitory_positive_mask = (self.rec_w.weight.data > 0) & inhibitory_mask

        #     self.rec_w.weight.data[excitatory_negative_mask] = 0
        #     self.rec_w.weight.data[inhibitory_positive_mask] = 0
        # else:
        #     excitatory_negative_mask = (self.fc_weights.weight.data < 0) & excitatory_mask
        #     inhibitory_positive_mask = (self.fc_weights.weight.data > 0) & inhibitory_mask

        #     self.fc_weights.weight.data[excitatory_negative_mask] = 0
        #     self.fc_weights.weight.data[inhibitory_positive_mask] = 0

        # self.fc_weights.weight.data[self.neuron_mask] = self.fc_weights.weight.data[self.neuron_mask].clamp(min=0)  # Excitatory weights ≥ 0
        # self.fc_weights.weight.data[~self.neuron_mask] = self.fc_weights.weight.data[~self.neuron_mask].clamp(max=0) 

        self.fc_weights.clip_weights()

        # print('inhibitory mask', inhibitory_mask.shape)

        # weights[weights < 0] *= excitatory_mask  # Set negative weights to 0 for excitatory neurons
        # weights[weights > 0] *= inhibitory_mask  # Set positive weights to 0 for inhibitory neurons



class OutputLayer(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            is_fc: bool,
            tau_fixed=None,
            bias = True,
            dt=0.5
    ):
        """
        output layer class
        :param is_fc: whether integrator is fc to r_out in rec or not
        """
        super(OutputLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_fc = is_fc
        self.dt = dt

        if is_fc:
            self.fc = DaleLinearLayer(in_dim, out_dim, bias=bias)
            if bias:
                nn.init.constant_(self.fc.bias, 0)
            nn.init.xavier_uniform_(self.fc.weight)

        # tau_m
        if tau_fixed is None:
            self.tau_m = nn.Parameter(torch.Tensor(out_dim))
            nn.init.constant_(self.tau_m, 5)
        else:
            self.tau_m = nn.Parameter(torch.Tensor(out_dim), requires_grad=False)
            nn.init.constant_(self.tau_m, tau_fixed)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_t, mem_t):
        """
        integrator neuron without spikes
        """
        alpha = torch.exp(-self.dt/self.tau_m)
        # alpha = self.sigmoid(self.tau_m)

        if self.is_fc:
            x_t = self.fc(x_t)
        else:
            x_t = x_t.view(-1, 10, int(self.in_dim / 10)).mean(dim=2)  # sum up population spike

        # d_mem = -soma_t + x_t
        mem = (mem_t + x_t) * alpha
        # mem = alpha * soma_t + (1 - alpha) * x_t
        return mem
    
    def clip_weights(self):
        """Clip weights based on excitatory/inhibitory mask."""
        self.fc.clip_weights()



# 2 hidden layers
class Decorrelation(nn.Module):
    def __init__(self):
        super(Decorrelation, self).__init__()
        self.decorr_matrix_next = None
    
    def forward(self, input, decorr_matrix_prev_batch):
        n=1e-3
        diag = torch.diag_embed(torch.square(input)) # (batch_size,hidden_dim,hidden_dim)

        input = input.reshape(input.shape[0],input.shape[1],1) # (batch_size,hidden_dim,1)
        input = torch.matmul(decorr_matrix_prev_batch, input) # (batch_size,hidden_dim,1)

        mult = torch.matmul(input, torch.transpose(input,1,2)) # (batch_size,hidden_dim,hidden_dim)
        update = torch.mean(mult - diag, dim=0) # (hidden_dim,hidden_dim)
        self.decorr_matrix_next = decorr_matrix_prev_batch - n * torch.matmul(update, decorr_matrix_prev_batch) # (hidden_dim,hidden_dim)

        input = input.reshape(input.shape[0],input.shape[1]) # (batch_size,hidden_dim)
        return input


class DaleLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, excitatory_ratio=0.75, device=None, dtype=None):
        """
        Dale's Law Linear Layer: Implements excitatory/inhibitory constraints.

        :param in_features: Number of input features
        :param out_features: Number of output features (neurons)
        :param bias: Whether to include a bias term
        :param excitatory_ratio: Fraction of excitatory neurons (default: 75% excitatory)
        :param device: Device for the tensors
        :param dtype: Data type for the tensors
        """
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        
        num_excitatory = int(0.75 * in_features)  # 75% excitatory neurons
        num_inhibitory = in_features - num_excitatory

        self.neuron_mask = torch.cat([
            torch.ones(num_excitatory, device=device),  # Excitatory neurons
            torch.zeros(num_inhibitory, device=device)  # Inhibitory neurons
        ]).bool()
        self.neuron_mask = self.neuron_mask[torch.randperm(self.neuron_mask.size(0))]  # Shuffle randomly
        self.neuron_mask = self.neuron_mask.unsqueeze(1).repeat(1, out_features).t()


    def clip_weights(self):
        """Clip weights based on excitatory/inhibitory mask."""

        # Apply masks: excitatory neurons (positive weights), inhibitory neurons (negative weights)
        self.weight.data[self.neuron_mask] = self.weight.data[self.neuron_mask].clamp(min=0)  # Excitatory weights ≥ 0
        self.weight.data[~self.neuron_mask] = self.weight.data[~self.neuron_mask].clamp(max=0)  # Inhibitory weights ≤ 0

        



class SnnNetwork(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate: float,
            is_rec: bool,
            rise_time: bool,
            act_fun_adp: ActFun_adp, #XXX
            device, #XXX
            b_j0=0.1, #XXX
            bias = True
    ):
        super(SnnNetwork, self).__init__()
        # is_adapt=True, one_to_one=True, dp_rate=DROPOUT_RATE, is_rec=False, rise_time=RISE_TIME, 
        # act_fun_adp=act_fun_adp, device=device, b_j0=B_J0
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one
        self.is_rec = is_rec
        self.rise_time = rise_time
        self.act_fun_adp = act_fun_adp #XXX
        self.device = device #XXX
        self.b_j0 = b_j0 #XXX

        self.dp = nn.Dropout(dp_rate)

        if self.rise_time:
            self.layer1 = SnnLayerRiseTime(hidden_dims[0], hidden_dims[0], is_rec=is_rec, is_adapt=is_adapt,
                               one_to_one=one_to_one, bias=bias, act_fun_adp=act_fun_adp, device=device)
        else:
            self.layer1 = SnnLayer(hidden_dims[0], hidden_dims[0], is_rec=is_rec, is_adapt=is_adapt,
                               one_to_one=one_to_one, bias=bias, act_fun_adp=act_fun_adp, device=device)

        # r in to r out
        self.layer1to2 = DaleLinearLayer(hidden_dims[0], hidden_dims[1], bias=bias)
        nn.init.xavier_uniform_(self.layer1to2.weight)

        # r out to r in
        self.layer2to1 = DaleLinearLayer(hidden_dims[1], hidden_dims[0], bias=bias)
        nn.init.xavier_uniform_(self.layer2to1.weight)

        if self.rise_time:
            self.layer2 = SnnLayerRiseTime(hidden_dims[1], hidden_dims[1], is_rec=is_rec, is_adapt=is_adapt,
                               one_to_one=one_to_one, bias=bias, act_fun_adp=act_fun_adp, device=device)
        else:
            self.layer2 = SnnLayer(hidden_dims[1], hidden_dims[1], is_rec=is_rec, is_adapt=is_adapt,
                               one_to_one=one_to_one, bias=bias, act_fun_adp=act_fun_adp, device=device)

        self.output_layer = OutputLayer(hidden_dims[1], out_dim, is_fc=True, bias=bias)

        self.out2layer2 = DaleLinearLayer(out_dim, hidden_dims[1], bias=bias)
        nn.init.xavier_uniform_(self.out2layer2.weight)

        if bias:
            nn.init.constant_(self.layer1to2.bias, 0)
            nn.init.constant_(self.layer2to1.bias, 0)
            nn.init.constant_(self.out2layer2.bias, 0)



        self.fr_layer2 = 0
        self.fr_layer1 = 0

        self.error1 = 0
        self.error2 = 0

    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t*0.5)
        # poisson
        # x_t = x_t.gt(0.7).float()

        soma_1, spk_1, a_curr_1, b_1 = self.layer1(ff=x_t, fb=self.layer2to1(h[5]), soma_t=h[0], spk_t=h[1],
                                                   a_curr_t=h[2], b_t=h[3])

        self.error1 = a_curr_1 - soma_1

        # use out mem signal as feedback
        soma_2, spk_2, a_curr_2, b_2 = self.layer2(ff=self.layer1to2(spk_1), fb=self.out2layer2(F.normalize(h[-1], dim=1)), soma_t=h[4],
                                                   spk_t=h[5], a_curr_t=h[6], b_t=h[7])

        self.error2 = a_curr_2 - soma_2

        self.fr_layer2 = self.fr_layer2 + spk_2.detach().cpu().numpy().mean()
        self.fr_layer1 = self.fr_layer1 + spk_1.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_2, h[-1])

        h = (soma_1, spk_1, a_curr_1, b_1,
             soma_2, spk_2, a_curr_2, b_2,
             mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def inference(self, x_t, h, T, bystep=None):
        """
        only called during inference
        :param x_t: input
        :param h: hidden states
        :param T: sequence length
        :param bystep: if true, then x_t is a sequence
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            if bystep is None:
                log_softmax, h = self.forward(x_t, h)
            else:
                log_softmax, h = self.forward(x_t[t], h)


            log_softmax_hist.append(log_softmax)
            h_hist.append(h)

        return log_softmax_hist, h_hist
    
    def inference_rise_time(self, x_t, h, T, bystep=None):
        """
        only called during inference
        :param x_t: input
        :param h: hidden states
        :param T: sequence length
        :param bystep: if true, then x_t is a sequence
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):

            if bystep is None:
                log_softmax, h = self.forward_rise_time(x_t, h)
            else:
                log_softmax, h = self.forward_rise_time(x_t[t], h)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h)
            
        return log_softmax_hist, h_hist

    def clamped_generate(self, test_class, zeros, h_clamped, T, clamp_value=0.5, batch=False, noise=None):
        """
        generate representations with mem of read out clamped
        :param test_class: which class is clamped
        :param zeros: input containing zeros, absence of input
        :param h: hidden states
        :param T: sequence length
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            if not batch:
                h_clamped[-1][0] = -clamp_value
                h_clamped[-1][0, test_class] = clamp_value
            else:
                h_clamped[-1][:, :] = torch.full(h_clamped[-1].size(), -clamp_value).to(self.device)
                h_clamped[-1][:, test_class] = clamp_value

            if noise is not None:
                    h_clamped[-1][:] += noise

            # if t==0:
            #     print(h_clamped[-1])

            log_softmax, h_clamped = self.forward(zeros, h_clamped)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h_clamped)

        return log_softmax_hist, h_hist
    
    def clamped_generate_rise_time(self, test_class, zeros, h_clamped, T, clamp_value=0.5, batch=False, noise=None):
        """
        generate representations with mem of read out clamped
        :param test_class: which class is clamped
        :param zeros: input containing zeros, absence of input
        :param h: hidden states
        :param T: sequence length
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            if not batch:
                h_clamped[-1][0] = -clamp_value
                h_clamped[-1][0, test_class] = clamp_value
            else:
                h_clamped[-1][:, :] = torch.full(h_clamped[-1].size(), -clamp_value).to(self.device)
                h_clamped[-1][:, test_class] = clamp_value

            if noise is not None:
                    h_clamped[-1][:] += noise

            # if t==0:
            #     print(h_clamped[-1])

            log_softmax, h_clamped = self.forward_rise_time(zeros, h_clamped)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h_clamped)

        return log_softmax_hist, h_hist

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # r
            weight.new(bsz, self.hidden_dims[0]).uniform_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(self.b_j0),
            # p
            weight.new(bsz, self.hidden_dims[1]).uniform_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(self.b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )
    
    def clip_all_weights(self):
        """Clip weights of all layers based on masks."""
        self.layer1.clip_weights()
        self.layer2.clip_weights()
        self.layer1to2.clip_weights()
        self.layer2to1.clip_weights()
        self.out2layer2.clip_weights()
        self.output_layer.clip_weights()




# 3 hidden layers

class SnnNetwork3Layer(SnnNetwork):
    def __init__(
            self,
            in_dim: int,
            hidden_dims: list,
            out_dim: int,
            is_adapt: bool,
            one_to_one: bool,
            dp_rate: float,
            is_rec: bool,
            rise_time: bool,
            act_fun_adp: torch.autograd.Function, #XXX
            device, #XXX
            bias = True,
            b_j0=0.1 #XXX
    ):
        super().__init__(in_dim, hidden_dims, out_dim, is_adapt, one_to_one, dp_rate, is_rec, rise_time, act_fun_adp, device)
        self.b_j0 = b_j0 #XXX
        self.device = device #XXX
        self.act_fun_adp = act_fun_adp #XXX


        # decorrelation
        self.decorr_layer_0 = Decorrelation()
        self.decorr_layer_1 = Decorrelation()
        self.decorr_layer_2 = Decorrelation()
        self.decorr_layer_3 = Decorrelation()
        self.decorr_layer_4 = Decorrelation()

        if self.rise_time:
            self.layer3 = SnnLayerRiseTime(hidden_dims[2], hidden_dims[2], is_rec=is_rec, is_adapt=is_adapt,
                               one_to_one=one_to_one, bias=bias, act_fun_adp=act_fun_adp, device=device)
        else:
            self.layer3 = SnnLayer(hidden_dims[2], hidden_dims[2], is_rec=is_rec, is_adapt=is_adapt,
                               one_to_one=one_to_one, bias=bias, act_fun_adp=act_fun_adp, device=device)

        self.layer2to3 = DaleLinearLayer(hidden_dims[1], hidden_dims[2], bias=bias)
        nn.init.xavier_uniform_(self.layer2to3.weight)

        # r out to r in
        self.layer3to2 = DaleLinearLayer(hidden_dims[2], hidden_dims[1], bias=bias)
        nn.init.xavier_uniform_(self.layer3to2.weight)

        self.output_layer = OutputLayer(hidden_dims[2], out_dim, is_fc=True)

        self.out2layer3 = DaleLinearLayer(out_dim, hidden_dims[2], bias=bias)
        nn.init.xavier_uniform_(self.out2layer3.weight)

        self.fr_layer3 = 0

        self.error3 = 0

        self.input_fc = DaleLinearLayer(in_dim, hidden_dims[0], bias=bias)
        nn.init.xavier_uniform_(self.input_fc.weight)

        if bias:
            nn.init.constant_(self.layer2to3.bias, 0)
            nn.init.constant_(self.layer3to2.bias, 0)
            nn.init.constant_(self.out2layer3.bias, 0)
            nn.init.constant_(self.input_fc.bias, 0)
            print('bias set to 0')

    def forward_rise_time(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t)
        # poisson
        # x_t = x_t.gt(0.7).float()
        x_t = self.input_fc(x_t)

        soma_1, spk_1, a_curr_1, current_curr_1, b_1 = self.layer1(ff=x_t, fb=self.layer2to1(h[6]), soma_t=h[0], spk_t=h[1],
                                                   a_curr_t=h[2], current_curr_t=h[3], b_t=h[4])

        self.error1 = a_curr_1 - soma_1

        # use out mem signal as feedback
        soma_2, spk_2, a_curr_2, current_curr_2, b_2 = self.layer2(ff=self.layer1to2(spk_1), fb=self.layer3to2(h[11]), soma_t=h[5],
                                                   spk_t=h[6], a_curr_t=h[7], current_curr_t=h[8], b_t=h[9])

        self.error2 = a_curr_2 - soma_2

        soma_3, spk_3, a_curr_3, current_curr_3, b_3 = self.layer3(ff=self.layer2to3(spk_2), fb=self.out2layer3(F.normalize(h[-1], dim=1)), soma_t=h[10],
                                                   spk_t=h[11], a_curr_t=h[12], current_curr_t=h[13], b_t=h[14])
        # soma_3, spk_3, a_curr_3, b_3 = self.layer3(ff=self.layer2to3(spk_2), fb=0, soma_t=h[8],
        #                                            spk_t=h[9], a_curr_t=h[10], b_t=h[11])

        self.error3 = a_curr_3 - soma_3

        self.fr_layer3 = self.fr_layer3 + spk_3.detach().cpu().numpy().mean()
        self.fr_layer2 = self.fr_layer2 + spk_2.detach().cpu().numpy().mean()
        self.fr_layer1 = self.fr_layer1 + spk_1.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_3, h[-1])

        h = (soma_1, spk_1, a_curr_1, current_curr_1, b_1,
             soma_2, spk_2, a_curr_2, current_curr_2, b_2,
             soma_3, spk_3, a_curr_3, current_curr_3, b_3,
             mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h
    
    def forward(self, x_t, h):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t)
        # poisson
        # x_t = x_t.gt(0.7).float()
        x_t = self.input_fc(x_t)

        soma_1, spk_1, a_curr_1, b_1 = self.layer1(ff=x_t, fb=self.layer2to1(h[5]), soma_t=h[0], spk_t=h[1],
                                                   a_curr_t=h[2], b_t=h[3])

        self.error1 = a_curr_1 - soma_1

        # use out mem signal as feedback
        soma_2, spk_2, a_curr_2, b_2 = self.layer2(ff=self.layer1to2(spk_1), fb=self.layer3to2(h[9]), soma_t=h[4],
                                                   spk_t=h[5], a_curr_t=h[6], b_t=h[7])

        self.error2 = a_curr_2 - soma_2

        soma_3, spk_3, a_curr_3, b_3 = self.layer3(ff=self.layer2to3(spk_2), fb=self.out2layer3(F.normalize(h[-1], dim=1)), soma_t=h[8],
                                                   spk_t=h[9], a_curr_t=h[10], b_t=h[11])
        # soma_3, spk_3, a_curr_3, b_3 = self.layer3(ff=self.layer2to3(spk_2), fb=0, soma_t=h[8],
        #                                            spk_t=h[9], a_curr_t=h[10], b_t=h[11])

        self.error3 = a_curr_3 - soma_3

        self.fr_layer3 = self.fr_layer3 + spk_3.detach().cpu().numpy().mean()
        self.fr_layer2 = self.fr_layer2 + spk_2.detach().cpu().numpy().mean()
        self.fr_layer1 = self.fr_layer1 + spk_1.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_3, h[-1])

        h = (soma_1, spk_1, a_curr_1, b_1,
             soma_2, spk_2, a_curr_2, b_2,
             soma_3, spk_3, a_curr_3, b_3,
             mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)

        return log_softmax, h

    def forward_decorrelation(self, x_t, h, decorr_matrix_0, decorr_matrix_1, decorr_matrix_2, decorr_matrix_3, decorr_matrix_4):
        batch_dim, input_size = x_t.shape

        x_t = x_t.reshape(batch_dim, input_size).float()
        x_t = self.dp(x_t)

        # poisson
        # x_t = x_t.gt(0.7).float()

        # decorrelate input
        x_t = self.decorr_layer_0(x_t, decorr_matrix_0)
        decorr_matrix_0 = self.decorr_layer_0.decorr_matrix_next.data.clone()
        x_t = self.input_fc(x_t)

        # decorrelate input to L1
        x_t = self.decorr_layer_1(x_t, decorr_matrix_1)
        decorr_matrix_1 = self.decorr_layer_1.decorr_matrix_next.data.clone()
        
        soma_1, spk_1, a_curr_1, b_1 = self.layer1(ff=x_t, fb=self.layer2to1(h[5]), soma_t=h[0], spk_t=h[1],
                                                   a_curr_t=h[2], b_t=h[3])
        self.error1 = a_curr_1 - soma_1

        # decorrelate input to L2
        spk_1 = self.decorr_layer_2(spk_1, decorr_matrix_2)
        decorr_matrix_2 = self.decorr_layer_2.decorr_matrix_next.data.clone()

        # use out mem signal as feedback
        soma_2, spk_2, a_curr_2, b_2 = self.layer2(ff=self.layer1to2(spk_1), fb=self.layer3to2(h[9]), soma_t=h[4],
                                                   spk_t=h[5], a_curr_t=h[6], b_t=h[7])
        self.error2 = a_curr_2 - soma_2

        # decorrelate input to L3
        spk_2 = self.decorr_layer_3(spk_2, decorr_matrix_3)
        decorr_matrix_3 = self.decorr_layer_3.decorr_matrix_next.data.clone()

        soma_3, spk_3, a_curr_3, b_3 = self.layer3(ff=self.layer2to3(spk_2), fb=self.out2layer3(F.normalize(h[-1], dim=1)), soma_t=h[8],
                                                   spk_t=h[9], a_curr_t=h[10], b_t=h[11])
        self.error3 = a_curr_3 - soma_3

        # decorrelate input to output layer
        spk_3 = self.decorr_layer_4(spk_3, decorr_matrix_4)
        decorr_matrix_4 = self.decorr_layer_4.decorr_matrix_next.data.clone()
        
        self.fr_layer3 = self.fr_layer3 + spk_3.detach().cpu().numpy().mean()
        self.fr_layer2 = self.fr_layer2 + spk_2.detach().cpu().numpy().mean()
        self.fr_layer1 = self.fr_layer1 + spk_1.detach().cpu().numpy().mean()

        # read out from r_out neurons
        mem_out = self.output_layer(spk_3, h[-1])

        h = (soma_1, spk_1, a_curr_1, b_1,
             soma_2, spk_2, a_curr_2, b_2,
             soma_3, spk_3, a_curr_3, b_3,
             mem_out)

        log_softmax = F.log_softmax(mem_out, dim=1)
        return log_softmax, h, decorr_matrix_0, decorr_matrix_1, decorr_matrix_2, decorr_matrix_3, decorr_matrix_4

    def inference_decorrelation(self, x_t, h, T, decorr_matrix_0, decorr_matrix_1, decorr_matrix_2, decorr_matrix_3, decorr_matrix_4, bystep=None):
        """
        only called during inference
        :param x_t: input
        :param h: hidden states
        :param T: sequence length
        :param bystep: if true, then x_t is a sequence
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):

            if bystep is None:
                log_softmax, h, decorr_matrix_0, decorr_matrix_1, decorr_matrix_2, decorr_matrix_3, decorr_matrix_4 = self.forward_decorrelation(x_t, h, decorr_matrix_0, decorr_matrix_1, decorr_matrix_2, decorr_matrix_3, decorr_matrix_4)
            else:
                log_softmax, h, decorr_matrix_0, decorr_matrix_1, decorr_matrix_2, decorr_matrix_3, decorr_matrix_4 = self.forward_decorrelation(x_t[t], h, decorr_matrix_0, decorr_matrix_1, decorr_matrix_2, decorr_matrix_3, decorr_matrix_4)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h)
            
        return log_softmax_hist, h_hist

    def init_hidden_rise_time(self, bsz):
        weight = next(self.parameters()).data

        return (
            # l1
            weight.new(bsz, self.hidden_dims[0]).uniform_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(self.b_j0),
            # l2
            weight.new(bsz, self.hidden_dims[1]).uniform_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(self.b_j0),
            # l3
            weight.new(bsz, self.hidden_dims[2]).uniform_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).fill_(self.b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (
            # l1
            weight.new(bsz, self.hidden_dims[0]).uniform_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(self.b_j0),
            # l2
            weight.new(bsz, self.hidden_dims[1]).uniform_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(self.b_j0),
            # l3
            weight.new(bsz, self.hidden_dims[2]).uniform_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).fill_(self.b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )

    def init_hidden_allzero(self, bsz):
        weight = next(self.parameters()).data
        return (
            # l1
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).zero_(),
            weight.new(bsz, self.hidden_dims[0]).fill_(self.b_j0),
            # l2
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).zero_(),
            weight.new(bsz, self.hidden_dims[1]).fill_(self.b_j0),
            # l3
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).zero_(),
            weight.new(bsz, self.hidden_dims[2]).fill_(self.b_j0),
            # layer out
            weight.new(bsz, self.out_dim).zero_(),
            # sum spike
            weight.new(bsz, self.out_dim).zero_(),
        )

    def clamp_withnoise(self, test_class, zeros, h_clamped, T, noise, index, batch=False, clamp_value=0.5):
        """
        generate representations with mem of read out clamped
        :param test_class: which class is clamped
        :param zeros: input containing zeros, absence of input
        :param h: hidden states
        :param T: sequence length
        :param noise: noise values
        :param index: index in h where noise is added to
        :return:
        """

        log_softmax_hist = []
        h_hist = []

        for t in range(T):
            if not batch:
                h_clamped[-1][0] = -clamp_value
                h_clamped[-1][0, test_class] = clamp_value
            else:
                h_clamped[-1][:, :] = torch.full(h_clamped[-1].size(), -clamp_value).to(self.device)
                h_clamped[-1][:, test_class] = clamp_value

            if noise is not None:
                h_clamped[index][:, :] += noise * h_clamped[index][:, :]

            # if t==0:
            #     print(h_clamped[-1])

            log_softmax, h_clamped = self.forward(zeros, h_clamped)

            log_softmax_hist.append(log_softmax)
            h_hist.append(h_clamped)

        return log_softmax_hist, h_hist
    
    def clip_all_weights(self):
        """Clip weights of all layers based on masks."""
        self.layer1.clip_weights()
        self.layer2.clip_weights()
        self.layer3.clip_weights()
        self.layer1to2.clip_weights()
        self.layer2to1.clip_weights()
        self.layer2to3.clip_weights()
        self.layer3to2.clip_weights()
        self.out2layer3.clip_weights()
        self.output_layer.clip_weights()
        self.input_fc.clip_weights()





# test function
def test(model, test_loader, time_steps, device):
    model.eval()
    test_loss = 0
    correct = 0
    test_energy = 0


    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, model.in_dim)

        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(data.size(0))

            zero_weights = sum((param == 0).sum().item() for param in model.parameters() if param.requires_grad)
            print(f'Number of zero before inference: {zero_weights}')


            log_softmax_outputs, hidden = model.inference(data, hidden, time_steps) # TODO

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()

            pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]

            test_energy += (torch.sum(torch.abs(model.error1)) + torch.sum(torch.abs(model.error2)) + torch.sum(torch.abs(model.error3))) / target.size()[0] / sum(model.hidden_dims)

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    # wandb.log({'spike sequence': plot_spiking_sequence(hidden, target)})

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    test_energy /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_loss, 100. * correct / len(test_loader.dataset), test_energy



# test function
def test_decorrelation(model, test_loader, time_steps, decorr_matrix_0, decorr_matrix_1, 
                       decorr_matrix_2, decorr_matrix_3, decorr_matrix_4, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, model.in_dim)

        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(data.size(0))
            
            log_softmax_outputs, hidden = model.inference_decorrelation(data, hidden, time_steps, decorr_matrix_0, decorr_matrix_1, decorr_matrix_2, decorr_matrix_3, decorr_matrix_4)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()

            pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    # wandb.log({'spike sequence': plot_spiking_sequence(hidden, target)})

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_loss, 100. * correct / len(test_loader.dataset)






############################
#   Rise time functions   #
############################

class SnnLayerRiseTime(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            is_rec: bool,
            is_adapt: bool,
            one_to_one: bool,
            act_fun_adp: ActFun_adp, #XXX
            device, #XXX
            tau_m_init=15.,
            tau_curr_decay_init=10.,
            tau_adap_init=20,
            tau_a_init=15.,
            dt = 0.5,
            bias = True
    ):
        super(SnnLayerRiseTime, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.is_rec = is_rec
        self.is_adapt = is_adapt
        self.one_to_one = one_to_one
        self.dt = dt
        self.device = device #XXX
        self.act_fun_adp = act_fun_adp #XXX

        if is_rec:
            self.rec_w = nn.Linear(hidden_dim, hidden_dim, bias=bias)
            # init weights
            if bias:
                nn.init.constant_(self.rec_w.bias, 0)
            nn.init.xavier_uniform_(self.rec_w.weight)

            p = torch.full(self.rec_w.weight.size(), fill_value=0.5).to(device)
            self.weight_mask = torch.bernoulli(p)

        else:
            self.fc_weights = nn.Linear(in_dim, hidden_dim, bias=bias)
            if bias:
                nn.init.constant_(self.fc_weights.bias, 0)
            nn.init.xavier_uniform_(self.fc_weights.weight)

        # define param for time constants
        self.tau_adp = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_m = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_curr_decay = nn.Parameter(torch.Tensor(hidden_dim))
        self.tau_a = nn.Parameter(torch.Tensor(hidden_dim))

        nn.init.normal_(self.tau_adp, tau_adap_init, .1)
        nn.init.normal_(self.tau_m, tau_m_init, .1)
        nn.init.normal_(self.tau_curr_decay, tau_curr_decay_init, .1)
        nn.init.normal_(self.tau_a, tau_a_init, .1)

        # self.tau_adp = nn.Parameter(torch.Tensor(1))
        # self.tau_m = nn.Parameter(torch.Tensor(1))
        # self.tau_a = nn.Parameter(torch.Tensor(1))

        # nn.init.constant_(self.tau_adp, tau_adap_init)
        # nn.init.constant_(self.tau_m, tau_m_init)
        # nn.init.constant_(self.tau_a, tau_a_init)

        # nn.init.normal_(self.tau_adp, 200., 20.)
        # nn.init.normal_(self.tau_m, 20., .5)

        self.sigmoid = nn.Sigmoid()

    def mem_update(self, ff, fb, soma, spike, a_curr, current_curr, b, is_adapt, baseline_thre=0.1, r_m=3):
        """
        mem update for each layer of neurons
        :param ff: feedforward signal
        :param fb: feedback signal to apical tuft
        :param soma: mem voltage potential at soma
        :param spike: spiking at last time step
        :param a_curr: apical tuft current at last t
        :param current: input current at last t
        :param b: adaptive threshold
        :return:
        """
        # alpha = self.sigmoid(self.tau_m)
        # rho = self.sigmoid(self.tau_adp)
        # eta = self.sigmoid(self.tau_a)
        alpha = torch.exp(-self.dt/self.tau_m)
        current_decay = torch.exp(-self.dt/self.tau_curr_decay)
        rho = torch.exp(-self.dt/self.tau_adp)
        eta = torch.exp(-self.dt/self.tau_a)

        if is_adapt:
            beta = 1.8
        else:
            beta = 0.
                
        b = rho * b + (1 - rho) * spike  # adaptive contribution
        new_thre = baseline_thre + beta * b  # udpated threshold
        
        current_new = current_decay * current_curr + ff

        a_new = eta * a_curr + fb  # fb into apical tuft

        #print("mem update",current_decay , current_curr , ff, eta , a_curr , fb)
        
        soma_new = alpha * soma + shifted_sigmoid(a_new) + current_new - new_thre * spike
        # soma_new = alpha * soma + shifted_sigmoid(a_new) + rise * ff - new_thre * spike
        # soma_new = alpha * soma + 1/2 * (a_new) + ffs - new_thre * spike

        inputs_ = soma_new - new_thre

        spike = self.act_fun_adp(inputs_)  # act_fun : approximation firing function
        # mem = (1 - spike) * mem

        return soma_new, spike, a_new, current_new, new_thre, b

    def forward(self, ff, fb, soma_t, spk_t, a_curr_t, current_curr_t, b_t):
        """
        forward function of a single layer. given previous neuron states and current input, update neuron states

        :param ff: ff signal (not counting rec)
        :param fb: fb top down signal
        :param soma_t: soma voltage
        :param a_curr_t: apical tuft voltage
        :return:
        """

        if self.is_rec:
            self.rec_w.weight.data = self.rec_w.weight.data * self.weight_mask
            # self.rec_w.weight.data = (self.rec_w.weight.data < 0).float() * self.rec_w.weight.data
            r_in = ff + self.rec_w(spk_t)
        else:
            if self.one_to_one:
                r_in = ff
            else:
                r_in = self.fc_weights(ff)

        soma_t1, spk_t1, a_curr_t1, current_curr_t1, _, b_t1 = self.mem_update(r_in, fb, soma_t, spk_t, a_curr_t, current_curr_t, b_t, self.is_adapt)

        return soma_t1, spk_t1, a_curr_t1, current_curr_t1, b_t1
    



def test_rise_time(model, test_loader, time_steps, device):
    model.eval()
    test_loss = 0
    correct = 0
    test_energy = 0
    
    # for data, target in test_loader:
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, model.in_dim)

        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden_rise_time(data.size(0))
            
            log_softmax_outputs, hidden = model.inference_rise_time(data, hidden, time_steps)

            test_loss += F.nll_loss(log_softmax_outputs[-1], target, reduction='sum').data.item()

            pred = log_softmax_outputs[-1].data.max(1, keepdim=True)[1]

            test_energy += (torch.sum(torch.abs(model.error1)) + torch.sum(torch.abs(model.error2)) + torch.sum(torch.abs(model.error3))) / target.size()[0] / sum(model.hidden_dims)

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()

    # wandb.log({'spike sequence': plot_spiking_sequence(hidden, target)})

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    test_energy /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))

    return test_loss, 100. * correct / len(test_loader.dataset), test_energy


