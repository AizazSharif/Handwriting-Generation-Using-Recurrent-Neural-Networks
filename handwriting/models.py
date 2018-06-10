from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Named tuple to hold Gaussian Mixture parameters
MDNParams = namedtuple('MDNParams', ['mu1', 'mu2', 'log_sigma1', 'log_sigma2', 'rho', 'pi_logit'])


class HandwritingRNN(nn.Module):
    """Class for unconditional Handwriting generation

    Parameters:
        input_dim (int): the last dimension of a (seq_len, batch_size, input_dim) input.
        hidden_dim (int): the desired dimension of the hidden state.
        output_dim (int): the last dimension of a (seq_len, batch_size, output_dim) output.
        layer_type (str): type of recurrent layer.
        num_layers (int): number of layers.
        recurrent_dropout (float): dropout applied to recurrent layers.
        n_gaussian (int): number of gaussian mixture components.

    Returns:
        model (nn.Model): a custom pytorch model
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 layer_type, num_layers,
                 recurrent_dropout, n_gaussian):
        super(HandwritingRNN, self).__init__()

        # Params
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.recurrent_dropout = recurrent_dropout
        self.n_gaussian = n_gaussian

        if self.layer_type == "gru":
            model = nn.GRU
        elif self.layer_type == "lstm":
            model = nn.LSTM

        # Layers / nn objects
        self.rnn_layer = model(input_dim,
                             hidden_dim,
                             num_layers=num_layers,
                             dropout=self.recurrent_dropout,
                             batch_first=True)
        self.mdn_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, onehot=None):

        out, hidden = self.rnn_layer(input, hidden)

        # out is (batch_size, seq_len, hidden_dim * num_directions)
        # We now map out to the parameters of the Gaussian Mixture Model of https://arxiv.org/pdf/1308.0850.pdf

        # Flatten model output
        out = out.contiguous().view(-1, out.size(-1))
        # Obtain MDN parameters
        out = self.mdn_layer(out)

        mu1, mu2, log_sigma1, log_sigma2, rho, pi_logit, e_logit = out.split(self.n_gaussian, dim=1)
        # Store Gaussian Mixture params in a namedtuple
        mdnparams = MDNParams(mu1=mu1,
                              mu2=mu2,
                              log_sigma1=log_sigma1,
                              log_sigma2=log_sigma2,
                              rho=nn.functional.tanh(rho),
                              pi_logit=pi_logit)

        return mdnparams, e_logit, hidden

    def initHidden(self, batch_size):

        if self.layer_type == "gru":
            hidden_state = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))

            return hidden_state

        elif self.layer_type == "lstm":
            hidden_state = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))
            cell_state = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim))

            return hidden_state, cell_state


class ConditionalHandwritingRNN(nn.Module):
    """Class for conditional Handwriting generation

    Parameters:
        input_dim (int): the last dimension of a (seq_len, batch_size, input_dim) input
        hidden_dim (int): the desired dimension of the hidden state
        output_dim (int): the last dimension of a (seq_len, batch_size, output_dim) output
        layer_type (str): type of recurrent layer
        num_layers (int): number of layers
        recurrent_dropout (float): dropout applied to recurrent layers
        n_gaussian (int): number of gaussian mixture components
        n_window (int): number of gaussian components for the attention window
        onehot_dim (int): dimension of onehot encoding (=vocabulary size)

    Returns:
        model (nn.Model): a custom pytorch model
    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 layer_type, num_layers, recurrent_dropout,
                 n_gaussian, n_window, onehot_dim):
        super(ConditionalHandwritingRNN, self).__init__()

        # Params
        self.input_dim = input_dim
        self.layer_type = layer_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.recurrent_dropout = recurrent_dropout
        self.n_gaussian = n_gaussian
        self.n_window = n_window
        self.onehot_dim = onehot_dim

        if self.layer_type == "gru":
            model = nn.GRU
        elif self.layer_type == "lstm":
            model = nn.LSTM

        #########################
        # Layers definition
        #########################

        model0_input_dim = self.input_dim
        modelx_input_dim = self.hidden_dim + self.input_dim + self.onehot_dim

        # 1) Use batch first because we use torch.bmm afterwards. Without batch first
        # we would have to transpose the tensor which may lead to confusion
        # 2) Explicitly split into num_layers layers so that we can use skip connections
        self.rnn_layer0 = model(model0_input_dim, self.hidden_dim, batch_first=True)

        for k in range(1, self.num_layers):
            setattr(self, "rnn_layer%s" % k, model(modelx_input_dim, self.hidden_dim, batch_first=True))

        self.window_layer = nn.Linear(hidden_dim, 3 * self.n_window)
        self.mdn_layer = nn.Linear(hidden_dim, 1 + 6 * self.n_gaussian)

        list_layers = [getattr(self, "rnn_layer%s" % k) for k in range(self.num_layers)]
        list_layers += [self.window_layer, self.mdn_layer]

        #########################
        # Custom initialization
        #########################
        for layer in list_layers:
            for p_name, p in layer.named_parameters():
                if "weight" in p_name:
                    # Graves-like initialization
                    # (w/o the truncation which does not have much influence on the results)
                    nn.init.normal(p, mean=0, std=0.075)
        for p_name, p in self.window_layer.named_parameters():
            if "bias" in p_name:
                # Custom initi for bias so that the kappas do not grow too fast
                # and prevent sequence alignment
                nn.init.normal(p, mean=-4.0, std=0.1)

    def forward(self, x_input, hidden, onehot, training=True, running_kappa=None):

        # Initialize U to compute the gaussian windows
        U = Variable(torch.arange(0, onehot.size(1)), requires_grad=False)
        U = U.view(1, 1, 1, -1)  # prepare for broadcasting

        # Pass input to first layer
        out0, hidden[0] = self.rnn_layer0(x_input, hidden[0])
        # Compute the gaussian window parameters
        alpha, beta, kappa = torch.exp(self.window_layer(out0)).unsqueeze(-1).split(self.n_window, -2)
        #print (len(alpha), " alpha")
        #print (len(beta), " beta")
        
        # In training mode compute running_kappa = cumulative sum of kappa
        if training:
            running_kappa = kappa.cumsum(1)
        # Otherwise, update the previous kappa
        else:
            assert running_kappa is not None
            running_kappa = running_kappa.unsqueeze(1) + kappa
        #print (len(kappa), " kappa")
        # Compute the window
        phi = alpha * torch.exp(-beta * (running_kappa - U).pow(2))
        phi = phi.sum(-2)
        #print (phi.shape, " phi")
        window = torch.matmul(phi, onehot)

        # Save the last window/phi/kappa for plotting
        self.window = window[:, -1, :]
        self.phi = phi[:, -1, :]
        self.new_kappa = running_kappa[:, -1, :, :]

        # Next model layers
        out = torch.cat([out0, window, x_input], -1)
        for i in range(1, self.num_layers):
            out, hidden[i] = self.rnn_layer1(out, hidden[i])
            if i != self.num_layers - 1:
                out = torch.cat([out, window, x_input], -1)

        # Flatten model output so that the same operation is applied to each time step
        out = out.contiguous().view(-1, out.size(-1))
        out = self.mdn_layer(out)

        pi_logit, mu1, mu2, log_sigma1, log_sigma2, rho, e_logit = torch.split(out, self.n_gaussian, 1)

        # Store Gaussian Mixture params in a namedtuple
        mdnparams = MDNParams(mu1=mu1,
                              mu2=mu2,
                              log_sigma1=log_sigma1,
                              log_sigma2=log_sigma2,
                              rho=F.tanh(rho),
                              pi_logit=pi_logit)

        return mdnparams, e_logit, hidden

    def initHidden(self, batch_size):

        if self.layer_type == "gru":

            return [Variable(torch.zeros(1, batch_size, self.hidden_dim)) for i in range(self.num_layers)]

        elif self.layer_type == "lstm":


            return [[Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                         Variable(torch.zeros(1, batch_size, self.hidden_dim))] for i in range(self.num_layers)]
