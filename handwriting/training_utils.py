import numpy as np
from collections import namedtuple
from . import models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Named tuple to hold the data
DataContainer = namedtuple("DataContainer", ["strokes", "texts", "onehots"])


def process_text_data(conf, texts):
    """Process text data for conditional generation

    Strategy : replace all non literal characters by a dummy `#` token.
    One hot encode the str text on a character basis for conditional generation
    Keep a dictionary to convert onehot to str and vice versa

    Input:
        conf : custom class to hold hyperparams
        texts (list): the sequences in str format

    Returns:
        conf : custom class to hold hyperparams
        texts (list): the sequences in str format
        texts_one_hot (list): the sequences, onehot encoded
    """

    # Replace special characters with the # token to specify <unknown>
    texts = [l.rstrip()
              .replace("!", "#")
              .replace("?", "#")
              .replace(":", "#")
              .replace(",", "#")
              .replace(".", "#")
              .replace(";", "#")
              .replace("(", "#")
              .replace(")", "#")
              .replace("#", "#")
              .replace("\'", "#")
              .replace("\"", "#")
              .replace("+", "#")
              .replace("-", "#")
              .replace("/", "#")
              .replace("0", "#")
              .replace("1", "#")
              .replace("2", "#")
              .replace("3", "#")
              .replace("4", "#")
              .replace("5", "#")
              .replace("6", "#")
              .replace("7", "#")
              .replace("8", "#")
              .replace("9", "#")
             for l in texts]

    # Get list of unique characters
    conf.alphabet = np.unique(list("".join(texts))).tolist()
    conf.n_alphabet = len(conf.alphabet)

    # Dict mapping unique characters to an index and vice versa
    conf.d_char_to_idx = {}
    conf.d_idx_to_char = {}
    for char_idx, char in enumerate(conf.alphabet):
        conf.d_char_to_idx[char] = char_idx
        conf.d_idx_to_char[char_idx] = char

    # One hot encode the sequences
    texts_one_hot = []
    for line in texts:
        # Split line into its individual characters
        line_chars = list(line)
        one_hot = np.zeros((len(line_chars), conf.n_alphabet), dtype=np.float32)
        # Fill the one hot encoding
        for i, char in enumerate(line_chars):
            one_hot[i, conf.d_char_to_idx[char]] = 1.0
        texts_one_hot.append(one_hot)

    return conf, texts, texts_one_hot


def load_data(conf, data_path="data", validate=False):
    """Load data to train model for handwriting generation
        Shuffle the data and split between training and validation

    Input:
        conf : custom class to hold hyperparams
        data_path (str): path to data
        validate (bool): whether we are in validation mode. This is used
        to add the char to idx and idx to char mappings to conf.

    Returns:
        data_container (DataContainer): custom class holding the data
    """

    # Load the array of strokes
    raw_strokes = np.load('/home/aizaz/Desktop/Handwriting-Generation-Project/data/strokes.npy' , encoding="latin1")
    # Load the list of sentences
    with open('/home/aizaz/Desktop/Handwriting-Generation-Project/data/sentences.txt' ) as f:
        raw_texts = f.readlines()

    # We will compute the mean ratio len_stroke / len_onehot
    stroke_counter, text_counter = 0, 0
    # We remove pairs of (stroke, text) where len(stroke) < conf.bptt
    strokes, texts = [], []
    for s, t in zip(raw_strokes, raw_texts):
        # Put strokes in a list, throw out those with length smaller than bptt + 1
        # recall bptt is the seq len through which we backpropagate
        # + 1 comes from the tagret which is offset by +1
        if s.shape[0] > conf.bptt + 1:
            strokes.append(s)
            texts.append(t)
            # Update our stroke and text counters
            stroke_counter += s.shape[0]
            text_counter += len(t)

    # Compute the mean ratio len_stroke / len_onehot (used in conditional generation)
    conf.stroke_onehot_ratio = int(stroke_counter / text_counter)

    # Further processing of the text data in conditional mode (character removing, onehot encoding)
    if conf.train_conditional or validate:
        conf, texts, onehots = process_text_data(conf, texts)

    # Shuffle for good measure
    rng_state = np.random.get_state()
    np.random.shuffle(strokes)

    if conf.train_unconditional:

        # No train/val split as the losses are not very indicative of quality
        # and we prefer validating on qualitative visual inspection
        data_container = DataContainer(strokes=strokes, texts=None, onehots=None)

        return data_container

    if conf.train_conditional:
        # Also shuffle the text and one hot sequence
        np.random.set_state(rng_state)
        np.random.shuffle(texts)
        np.random.set_state(rng_state)
        np.random.shuffle(onehots)

        # No train/val split as the losses are not very indicative of quality
        # and we prefer validating on qualitative visual inspection
        data_container = DataContainer(strokes=strokes, texts=texts, onehots=onehots)

        return data_container


def get_model(conf, input_dim, output_dim, onehot_dim=None):
    """Utility for model loading

    Args:
        conf (Experimentconf): custom class to hold hyperparams
        input_dim (int): the last dimension of a (seq_len, batch_size, input_dim) input
        output_dim (int): the last dimension of a (seq_len, batch_size, output_dim) output
        onehot_dim (int): text one hot encoding dimension (=vocabulary size)

    Returns:
        model (nn.Model): a custom pytorch model
    """

    if conf.train_unconditional:
        model = models.HandwritingRNN(input_dim,
                                    conf.hidden_dim,
                                    output_dim,
                                    conf.layer_type,
                                    conf.num_layers,
                                    conf.recurrent_dropout,
                                    conf.n_gaussian)

    elif conf.train_conditional:

        assert onehot_dim is not None

        model = models.ConditionalHandwritingRNN(input_dim,
                                               conf.hidden_dim,
                                               output_dim,
                                               conf.layer_type,
                                               conf.num_layers,
                                               conf.recurrent_dropout,
                                               conf.n_gaussian,
                                               conf.n_window,
                                               onehot_dim)
    print ("********** Model Summary *****************")
    print(model)
    print ("******************************************")
    return model


def get_optimizer(conf, model):
    """Utility for gradient descent optimizer loading

    Args:
        conf (Experimentconf): custom class to hold hyperparams
        model (nn.Model): the model to optimize

    Returns:
        optimizer (torch.optimizer): a pytorch optimizer
    """

    if conf.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=conf.learning_rate)
    else:
        assert conf.optimizer == "rmsprop"
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=conf.learning_rate)

    return optimizer


def get_random_unconditional_training_batch(conf, data):
    """Utility to load a random batch for unconditional training

    Args:
        conf(Experimentconf): custom class to hold hyperparams
        data (DataContainer): custom class holding the data

    Returns:
        X_tensor, Y_tensor (torch.Tensor): input and target tensor
    """

    strokes = data.strokes
    stroke_dim = strokes[0].shape[-1]

    # Initialize numpy arrays where we'll fill features and targets
    # This time, we format data as batch first (cf. models.ConditionalHandwritingRNN)
    X_npy = np.zeros((conf.batch_size, conf.bptt, stroke_dim), dtype=np.float32)
    Y_npy = np.zeros((conf.batch_size, conf.bptt, stroke_dim), dtype=np.float32)

    # Sample strokes randomly
    idxs = np.random.randint(0, len(strokes), conf.batch_size)

    for batch_idx, idx in enumerate(idxs):

        # Select the stroke
        stroke = strokes[idx]
        # `data augmentation` : select random substroke
        start = np.random.randint(0, stroke.shape[0] - conf.bptt - 1)
        X_npy[batch_idx, ...] = stroke[start: start + conf.bptt, :]
        Y_npy[batch_idx, ...] = stroke[start + 1: start + 1 + conf.bptt, :]

    X_tensor = torch.from_numpy(X_npy)
    Y_tensor = torch.from_numpy(Y_npy)

    # Move data to GPU if required and wrap to Autograd Variable

    # Wrap to Autograd Variable
    X_tensor = Variable(X_tensor)
    Y_tensor = Variable(Y_tensor)

    # Check tensor dimensions
    assert X_tensor.size(0) == conf.batch_size
    assert Y_tensor.size(0) == conf.batch_size

    assert X_tensor.size(1) == conf.bptt
    assert Y_tensor.size(1) == conf.bptt

    assert X_tensor.size(2) == 3
    assert Y_tensor.size(2) == 3

    return X_tensor, Y_tensor


def get_random_conditional_training_batch(conf, data):
    """Utility to load a random batch for conditional training

    Input:
        conf(Experimentconf): custom class to hold hyperparams
        data (DataContainer): custom class holding the data

    Returns:
        X_tensor, Y_tensor (torch.Tensor): feature and target tensor
                           plus onehot tensor for conditional generation
    """

    strokes = data.strokes
    onehots = data.onehots

    # Sanity check. Each sequence should have its corresponding one hot representation
    assert len(strokes) == len(onehots)

    strokes_dim = strokes[0].shape[-1]
    onehot_dim = onehots[0].shape[-1]

    # Having defined the stroke_onehot_ratio, we can compute the
    # approximate length of the onehot sequence corresponding to
    # conf.bptt steps in the stroke sequence
    # We do this to facilitate the learning of the alignment between
    # stroke and onehot.
    onehot_len = conf.bptt // conf.stroke_onehot_ratio

    # Initialize numpy arrays where we'll fill features and targets
    # We format data as batch first (cf. models.ConditionalHandwritingRNN for explanation)
    X_npy = np.zeros((conf.batch_size, conf.bptt, strokes_dim), dtype=np.float32)
    Y_npy = np.zeros((conf.batch_size, conf.bptt, strokes_dim), dtype=np.float32)
    onehot_npy = np.zeros((conf.batch_size, onehot_len, onehot_dim), dtype=np.float32)

    # Get the list of sequences corresponding to the batch
    idxs = np.random.randint(0, len(strokes), conf.batch_size)

    for batch_idx, idx in enumerate(idxs):
        stroke = strokes[idx]
        onehot = onehots[idx]

        # We only use the start of the stroke
        # Otherwise, the network would have to also learn where to
        # start the alignment
        X_npy[batch_idx, ...] = stroke[:conf.bptt]
        Y_npy[batch_idx, ...] = stroke[1: conf.bptt + 1]
        onehot_npy[batch_idx, :onehot.shape[0], :] = onehot[:onehot_len, :]

    X_tensor = torch.from_numpy(X_npy)
    Y_tensor = torch.from_numpy(Y_npy)
    onehot_tensor = torch.from_numpy(onehot_npy)

    # Wrap to Autograd Variable 
    X_tensor = Variable(X_tensor)
    Y_tensor = Variable(Y_tensor)
    onehot_tensor = Variable(onehot_tensor)

    # Check tensor dimensions
    assert X_tensor.size(0) == conf.batch_size
    assert Y_tensor.size(0) == conf.batch_size
    assert onehot_tensor.size(0) == conf.batch_size

    assert X_tensor.size(1) == conf.bptt
    assert Y_tensor.size(1) == conf.bptt

    assert X_tensor.size(2) == 3
    assert Y_tensor.size(2) == 3

    return X_tensor, Y_tensor, onehot_tensor


def train_step(conf, model, X_var, Y_var, optimizer, onehot=None):
    """Full training step (forward + backward pass + weight update)

    Input:
        conf : custom class to hold hyperparams
        model (nn.Model): the model to train
        X_var, Y_var (torch.Variable): the
        optimizer (torch.optimizer): the optimizer to train the model
        onehot (torch.Variable or None): the onehot encode text for conditional generation

    Returns:
        d_loss (dict): python dictionary storing training metrics
    """

    # Set NN to train mode (deals with dropout and batchnorm)
    model.train()

    # Reset gradients
    optimizer.zero_grad()

    # Initialize hidden
    hidden = model.initHidden(X_var.size(0))

    # Forward pass
    mdnparams, e_logit, _ = model(X_var, hidden, onehot=onehot)

    # Flatten target
    target = Y_var.view(-1, 3).contiguous()
    # Extract eos, X1, X2
    eos, X1, X2 = target.split(1, dim=1)

    # Compute nll loss for next stroke 2D prediction
    nll = gaussian_2Dnll(X1, X2, mdnparams)
    # Compute binary classification loss for end of sequence tag
    loss_bce = classification_loss(eos, e_logit)

    # Sum the losses
    total_loss = (nll + loss_bce)

    d_loss = {"nll": nll.data.cpu().numpy()[0],
              "bce": loss_bce.data.cpu().numpy()[0],
              "total": total_loss.data.cpu().numpy()[0]}

    # Backward pass
    total_loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm(model.parameters(), conf.gradient_clipping)
    optimizer.step()

    return d_loss


def classification_loss(e_truth, e_logit):
    """Compute binary cross entropy with logits between target and output logits

    Args:
        e_truth (Variable): target eos
        e_logit (Variable): predicted logits

    Returns:
        classification_loss (Variable): the binary cross entropy loss
    """

    classification_loss = nn.BCEWithLogitsLoss(size_average=True)(e_logit, e_truth)

    return classification_loss


def logsumexp(x):
    """Logsumexp trick to avoid overflow in a log of sum of exponential expression

    Args:
        x (Variable or Tensor): the input on which to compute the log of sum of exponential

    Returns:
        logsum (Variable or Tensor): the computed log of sum of exponential
    """

    assert x.dim() == 2
    x_max, x_max_idx = x.max(dim=-1, keepdim=True)
    logsum = x_max + torch.log((x - x_max).exp().sum(dim=-1, keepdim=True))
    return logsum


def compute_Z(X1, X2, mu1, mu2, log_sigma1, log_sigma2, rho):
    """Compute the z quantity of Formula 25 from https://arxiv.org/pdf/1308.0850.pdf

    Args:
        X1 (Variable or Tensor): specify where to evaluate 2D gaussian
        X2 (Variable or Tensor): specify where to evaluate 2D gaussian
        mu1 (Variable or Tensor): mean the first gaussian component
        mu2 (Variable or Tensor): mean the second gaussian component
        log_sigma1 (Variable or Tensor): log standard deviation of the first gaussian component
        log_sigma2 (Variable or Tensor): log standard deviation of the second gaussian component
        rho (Variable or Tensor): captures the correlation between the gaussian components

    Returns:
        Z (Variable or Tensor): the computed Z
    """

    # Formula 25 from https://arxiv.org/pdf/1308.0850.pdf
    term1 = torch.pow((X1 - mu1) / log_sigma1.exp(), 2)
    term2 = torch.pow((X2 - mu2) / log_sigma2.exp(), 2)
    term3 = -2 * rho * (X1 - mu1) * (X2 - mu2) / (log_sigma1.exp() * log_sigma2.exp())
    Z = term1 + term2 + term3

    return Z


def gaussian_2Dnll(X1, X2, mdnparams):
    """Compute 2D gaussian negative log-likelihood

    Args:
        X1 (Variable): the first component of the target stroke
        X2 (Variable): the second component of the target stroke
        mdnparams (namedtuple): holds the predicted Mixture Gaussian parameters

    Returns:
        nll (Variable): the computed 2D gaussian negative log-likelihood
    """

    # Roll out MDN params
    mu1 = mdnparams.mu1
    mu2 = mdnparams.mu2
    log_sigma1 = mdnparams.log_sigma1
    log_sigma2 = mdnparams.log_sigma2
    rho = mdnparams.rho
    pi_logit = mdnparams.pi_logit

    # Expand to the same size as the gaussian components
    X1 = X1.expand_as(mu1)
    X2 = X2.expand_as(mu1)

    Z = compute_Z(X1, X2, mu1, mu2, log_sigma1, log_sigma2, rho)

    # Rewrite likelihood part of Eq. 26 as logsumexp for stability
    pi_term = F.log_softmax(pi_logit)
    Z_term = -0.5 * Z / (1 - torch.pow(rho, 2))
    sigma_term = - torch.log(2 * float(np.pi) * log_sigma1.exp() * log_sigma2.exp() * torch.sqrt(1 - torch.pow(rho, 2)))

    exp_term = pi_term + Z_term + sigma_term
    nll = -logsumexp(exp_term).squeeze().mean()

    return nll
