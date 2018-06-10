import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from . import training_utils as t_units
from . import log as logs
from . import inference_utils as i_utils
from utils import visualization_utils as v_utils


def train_unconditional(conf):
    """Train model for unconditional handwriting generation
    # Input:
    #   configurations for unconditional training

    # Output:
    #   model trained and saved to disk
    """

    data = t_units.load_data(conf)

    # Model specifications
    input_dimensions = data.strokes[0].shape[-1]
    output_dimensions = 6 * conf.n_gaussian + 1
    model = t_units.get_model(conf, input_dimensions, output_dimensions)
    optimizer = t_units.get_optimizer(conf, model)

  
    loss = ""
    d_monitor = defaultdict(list)

    # ***************** Training *************************
    logs.print_red("Starting training")
    for epoch in tqdm(range(conf.nb_epoch), desc="Training"):

        # Track the training losses over an epoch
        d_epoch_monitor = defaultdict(list)

        # Loop over batches
        desc = "Epoch: %s -- %s" % (epoch, loss)
        for batch in tqdm(range(conf.n_batch_per_epoch), desc=desc):

            # Sample a batch (X, Y)
            X_var, Y_var = t_units.get_random_unconditional_training_batch(conf, data)

            # Train step = forward + backward + weight update
            d_loss = t_units.train_step(conf, model, X_var, Y_var, optimizer)


            d_epoch_monitor["bce"].append(d_loss["bce"])
            d_epoch_monitor["nll"].append(d_loss["nll"])
            d_epoch_monitor["total"].append(d_loss["total"])

        # Sample a sequence to follow progress and save the plot
        plot_data = i_utils.sample_unconditional_sequence(conf, model)
        v_utils.plot_stroke(plot_data.stroke, "Plots/unconditional_training/epoch_%s.png" % epoch)

        # Update d_monitor with the mean over an epoch
        for key in d_epoch_monitor.keys():
            d_monitor[key].append(np.mean(d_epoch_monitor[key]))
        # Prepare loss to update progress bar
        loss = "Total : %.3g " % (d_monitor["total"][-1])

        # Save the model at regular intervals
        if epoch % 5 == 0:


            # Move model to cpu before training to allow inference on cpu
            model.cpu()
            torch.save(model, conf.unconditional_model_path)


    logs.print_red("Finished training")


def train_conditional(conf):
    """Train model for conditional handwriting generation

    # Input:
    #   configurations for conditional training

    # Output:
    #   model trained and saved to disk
    """

    list_data_train = t_units.load_data(conf)

    # Model specifications
    input_size = list_data_train[0][0].shape[1]
    print ("Input Size : ",input_size)
    onehot_dim = list_data_train[-1][0].shape[-1]
    print ("Onehot dimensions : ",onehot_dim)
    output_size = 3 * conf.n_gaussian + 1
    print ("Output Size : ",output_size)
    model = t_units.get_model(conf, input_size, output_size, onehot_dim=onehot_dim)
    optimizer = t_units.get_optimizer(conf, model)

    loss = ""
    d_monitor = defaultdict(list)

    # ***************** Training *************************
    logs.print_red("Starting training")
    for epoch in tqdm(range(conf.nb_epoch), desc="Training"):

        # Track the training losses over an epoch
        d_epoch_monitor = defaultdict(list)

        # Loop over batches
        desc = "Epoch: %s -- %s" % (epoch, loss)
        for batch in tqdm(range(conf.n_batch_per_epoch), desc=desc):

            X_var, Y_var, onehot_var = t_units.get_random_conditional_training_batch(conf, list_data_train)
            #print (X_var.shape, " X_var")
            #print (Y_var.shape, " Y_var")
            #print (onehot_var, " onehot_var")

            # Train step.
            d_loss = t_units.train_step(conf, model, X_var, Y_var, optimizer, onehot=onehot_var)

            d_epoch_monitor["bce"].append(d_loss["bce"])
            d_epoch_monitor["nll"].append(d_loss["nll"])
            d_epoch_monitor["total"].append(d_loss["total"])

        # Update d_monitor with the mean over an epoch
        for key in d_epoch_monitor.keys():
            d_monitor[key].append(np.mean(d_epoch_monitor[key]))
        # Prepare loss to update progress bar
        loss = "Total : %.3g  " % (d_monitor["total"][-1])

        plot_data = i_utils.sample_fixed_sequence(conf, model)
        v_utils.plot_stroke(plot_data.stroke, "Plots/conditional_training/epoch_%s.png" % epoch)

        # Move model to cpu before training to allow inference on cpu
        if epoch % 5 == 0:


            # Move model to cpu before training to allow inference on cpu
            model.cpu()
            torch.save(model, conf.conditional_model_path)

    logs.print_red("Finished training")
