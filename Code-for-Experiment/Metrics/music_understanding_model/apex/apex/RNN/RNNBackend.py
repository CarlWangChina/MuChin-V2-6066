import torch
import Code_for_Experiment.Metrics.music_understanding_model.apex.apex.amp.rnn_compat
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn as nn

def is_iterable(maybe_iterable):
    return isinstance(maybe_iterable, list) or isinstance(maybe_iterable, tuple)

def flatten_list(tens_list):
    if not is_iterable(tens_list):
        return tens_list
    return torch.cat(tens_list, dim=0).view(len(tens_list), *tens_list[0].size())

class bidirectionalRNN(nn.Module):
    def __init__(self, inputRNN, num_layers=1, dropout=0):
        super(bidirectionalRNN, self).__init__()
        self.dropout = dropout
        self.fwd = stackedRNN(inputRNN, num_layers=num_layers, dropout=dropout)
        self.bckwrd = stackedRNN(inputRNN.new_like(), num_layers=num_layers, dropout=dropout)
        self.rnns = nn.ModuleList([self.fwd, self.bckwrd])

    def forward(self, input, collect_hidden=False):
        seq_len = input.size(0)
        bsz = input.size(1)
        fwd_out, fwd_hiddens = list(self.fwd(input, collect_hidden=collect_hidden))
        bckwrd_out, bckwrd_hiddens = list(self.bckwrd(input, reverse=True, collect_hidden=collect_hidden))
        output = torch.cat([fwd_out, bckwrd_out], -1)
        hiddens = tuple(torch.cat(hidden, -1) for hidden in zip(fwd_hiddens, bckwrd_hiddens))
        return output, hiddens

    def reset_parameters(self):
        for rnn in self.rnns:
            rnn.reset_parameters()

    def init_hidden(self, bsz):
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        for rnn in self.rnns:
            rnn.detach_hidden()

    def reset_hidden(self, bsz):
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):
        for rnn in self.rnns:
            rnn.init_inference(bsz)

class stackedRNN(nn.Module):
    def __init__(self, inputRNN, num_layers=1, dropout=0):
        super(stackedRNN, self).__init__()
        self.dropout = dropout
        if isinstance(inputRNN, RNNCell):
            self.rnns = [inputRNN]
            for i in range(num_layers - 1):
                self.rnns.append(inputRNN.new_like(inputRNN.output_size))
        elif isinstance(inputRNN, list):
            assert len(inputRNN) == num_layers, "RNN list length must be equal to num_layers"
            self.rnns = inputRNN
        else:
            raise RuntimeError()
        self.nLayers = len(self.rnns)
        self.rnns = nn.ModuleList(self.rnns)

    def forward(self, input, collect_hidden=False, reverse=False):
        seq_len = input.size(0)
        bsz = input.size(1)
        inp_iter = reversed(range(seq_len)) if reverse else range(seq_len)
        hidden_states = [[] for i in range(self.nLayers)]
        outputs = []
        for seq in inp_iter:
            for layer in range(self.nLayers):
                if layer == 0:
                    prev_out = input[seq]
                outs = self.rnns[layer](prev_out)
                if collect_hidden:
                    hidden_states[layer].append(outs)
                elif seq == seq_len - 1:
                    hidden_states[layer].append(outs)
                prev_out = outs[0]
            outputs.append(prev_out)
        if reverse:
            outputs = list(reversed(outputs))
        output = flatten_list(outputs)
        if not collect_hidden:
            seq_len = 1
        n_hid = self.rnns[0].n_hidden_states
        new_hidden = [[[None for k in range(self.nLayers)] for j in range(seq_len)] for i in range(n_hid)]
        for i in range(n_hid):
            for j in range(seq_len):
                for k in range(self.nLayers):
                    new_hidden[i][j][k] = hidden_states[k][j][i]
        hidden_states = new_hidden
        if reverse:
            hidden_states = list(list(reversed(list(entry))) for entry in hidden_states)
        hiddens = list(list(flatten_list(seq) for seq in hidden) for hidden in hidden_states)
        if not collect_hidden:
            hidden_states = list(entry[0] for entry in hidden_states)
        return output, hidden_states

    def reset_parameters(self):
        for rnn in self.rnns:
            rnn.reset_parameters()

    def init_hidden(self, bsz):
        for rnn in self.rnns:
            rnn.init_hidden(bsz)

    def detach_hidden(self):
        for rnn in self.rnns:
            rnn.detach_hidden()

    def reset_hidden(self, bsz):
        for rnn in self.rnns:
            rnn.reset_hidden(bsz)

    def init_inference(self, bsz):
        for rnn in self.rnns:
            rnn.init_inference(bsz)

class RNNCell(nn.Module):
    def __init__(self, gate_multiplier, input_size, hidden_size, cell, n_hidden_states=2, bias=False, output_size=None):
        super(RNNCell, self).__init__()
        self.gate_multiplier = gate_multiplier
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.bias = bias
        self.output_size = output_size
        if output_size is None:
            self.output_size = hidden_size
        self.gate_size = gate_multiplier * self.hidden_size
        self.n_hidden_states = n_hidden_states
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.output_size))
        if self.output_size != self.hidden_size:
            self.w_ho = nn.Parameter(torch.Tensor(self.output_size, self.hidden_size))
        self.b_ih = self.b_hh = None
        if self.bias:
            self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
            self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))
        self.hidden = [None for states in range(self.n_hidden_states)]
        self.reset_parameters()

    def new_like(self, new_input_size=None):
        if new_input_size is None:
            new_input_size = self.input_size
        return type(self)(self.gate_multiplier,
                           new_input_size,
                           self.hidden_size,
                           self.cell,
                           self.n_hidden_states,
                           self.bias,
                           self.output_size)

    def reset_parameters(self, gain=1):
        stdev = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-stdev, stdev)

    def init_hidden(self, bsz):
        for param in self.parameters():
            if param is not None:
                a_param = param
                break
        for i, _ in enumerate(self.hidden):
            if self.hidden[i] is None or self.hidden[i].data.size()[0] != bsz:
                if i == 0:
                    hidden_size = self.output_size
                else:
                    hidden_size = self.hidden_size
                tens = a_param.data.new(bsz, hidden_size).zero_()
                self.hidden[i] = Variable(tens, requires_grad=False)

    def reset_hidden(self, bsz):
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = None
        self.init_hidden(bsz)

    def detach_hidden(self):
        for i, _ in enumerate(self.hidden):
            if self.hidden[i] is None:
                raise RuntimeError("Must initialize hidden state before you can detach it")
        for i, _ in enumerate(self.hidden):
            self.hidden[i] = self.hidden[i].detach()

    def forward(self, input):
        self.init_hidden(input.size()[0])
        hidden_state = self.hidden[0] if self.n_hidden_states == 1 else self.hidden
        self.hidden = self.cell(input, hidden_state, self.w_ih, self.w_hh, b_ih=self.b_ih, b_hh=self.b_hh)
        if self.n_hidden_states > 1:
            self.hidden = list(self.hidden)
        else:
            self.hidden = [self.hidden]
        if self.output_size != self.hidden_size:
            self.hidden[0] = F.linear(self.hidden[0], self.w_ho)
        return tuple(self.hidden)