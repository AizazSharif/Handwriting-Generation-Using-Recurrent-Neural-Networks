import os
import torch

from . import log as logs
from . import training_utils as t_utils
from . import inference_utils as i_utils
from utils import visualization_utils as utils


def validate_unconditional(conf):
    """Validate model for unconditional handwriting generation

    """
    t_utils.load_data(conf, validate=True)

    if not os.path.isfile(conf.unconditional_model_path):
        logs.print_red("Unconditional model does not exist. Please train one first")

    # Load model
    model = torch.load(conf.unconditional_model_path)


    # Count figure number
    counter = 0
    for counter in range(10):

        # Sample a sequence to follow progress and save the plot
        plot_data = i_utils.sample_unconditional_sequence(conf, model)
        utils.plot_stroke(plot_data.stroke, "Plots/validation/unconditional_sample_%s.png" % counter)

    logs.print_red("Results saved to figures/validation")


def validate_conditional(conf):
    """Validate model for conditional handwriting generation

    """

    t_utils.load_data(conf, validate=True)

    if not os.path.isfile(conf.conditional_model_path):
        logs.print_red("Conditional model does not exist. Please train one first")

    # Load model
    model = torch.load(conf.conditional_model_path)

    # Count figure number
    counter = 0
    continue_flag = None
    while True:

        input_text = input("Enter text: ")

        # Check all characters are allowed
        for char in input_text:
            try:
                conf.d_char_to_idx[char]
            except KeyError:
                logs.print_red("%s not in alphabet" % char)
                continue_flag = True

        # Ask for a new input text in case of failogsre
        if continue_flag:
            continue

        plot_data = i_utils.sample_fixed_sequence(conf, model, truth_text=input_text)
        utils.plot_stroke(plot_data.stroke, "Plots/validation/conditional_sample_%s.png" % counter)
        logs.print_red("Results saved to figures/validation")

        counter += 1
