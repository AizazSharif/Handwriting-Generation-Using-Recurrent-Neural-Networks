import os
import shutil


class ExperimentSettings(object):
    """Custom class to hold experiment parameters

    Args:
        cli_args (dict or object with __dict__ method): command line arguments or dictionary

    """

    def __init__(self, cli_args):

        # Transfer attributes
        if isinstance(cli_args, dict):
            self.__dict__.update(cli_args)
        else:
            self.__dict__.update(cli_args.__dict__)

        # Load simulation and training settings and prepare directories
        self.setup_dir()

    def setup_dir(self):

        for path in ["figures",
                     "figures/unconditional_samples",
                     "figures/conditional_samples",
                     "figures/validation",
                     "data/processed",
                     "models"]:
            if not os.path.exists(path):
                os.makedirs(path)

        # Cleanup figures if training
        if hasattr(self, "train_unconditional"):
            if self.train_unconditional:

                shutil.rmtree("figures/unconditional_samples")
                os.makedirs("figures/unconditional_samples")

        if hasattr(self, "train_unconditional"):
            if self.train_conditional:

                shutil.rmtree("figures/conditional_samples")
                os.makedirs("figures/conditional_samples")

        self.fig_dir = "figures"
        self.data_dir = "data/processed"
        self.model_dir = "models"
