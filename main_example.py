# Description: Main file to run DeepMaxent model on the Elith data
import os
import warnings

from librairies.train_models_elith import cv_eval_deepmodel, eval_deepmodel
from librairies.utils import (
    get_random_seed_list,
    make_results_directory,
    set_seed,
)


class ConfigArgs:
    """Configuration arguments for DeepMaxent model."""

    def __init__(self, config_file=None):
        if config_file:
            self.load_configuration(config_file)
        else:
            self.load_default_configuration()

    def load_default_configuration(self):
        """Load default configuration parameters."""
        # General parameters
        self.outputdir = "results_deepmaxent"
        self.dirdata = "./data/elith/"
        self.global_seed = 42  # the answer to life the universe and everything
        set_seed(self.global_seed)
        self.list_of_seed = get_random_seed_list(1000)
        self.repeat_seed = 1
        
        # Cross-validation step
        self.cv = False  # True for cross-validation, False for evaluation
        self.lr_tested = [0.00002, 0.0002, 0.002]  # Learning rate tested
        self.batch_tested = [10, 250]  # Batch size tested
        self.hidden_nbr_tested = [1, 2]  # Number of hidden layers tested
        self.w_tested = [3e-4, 3e-3]  # Weight decay tested
        self.num_cv_blocks = (5, 5)  # Number of blocks for CV
        self.validation_size = 0.1  # Size of the validation set
        self.cross_validation = "blocked"  # "blocked" or "random"
        self.validation_criteria = "AUC"  # "AUC" or "loss"
        self.epoch_cv = 50  # Number of epochs for cross-validation
        
        # Final hyperparameter values
        self.learning_rate = 0.00002
        self.epoch = 100
        self.hidden_size = 250
        self.batch_size = 250
        self.weight_decay = 3e-4 #if mean in deepmaxentloss
        self.TGB = True  # True for using target group background correction
        self.hidden_nbr = 2
        self.loss_options = ["deepmaxent"] # "poisson", "deepmaxent", "bce", "ce"

        self.regions = ["CAN"]

        self.categoricalvars = ["ontveg", "vegsys"]



def disable_warnings():
    """Disable all warnings."""
    warnings.filterwarnings("ignore")


def run(args):
    """Run the DeepMaxent model."""
    disable_warnings()
    make_results_directory(args)
    set_seed(args.global_seed)
    if args.cv:
        cv_eval_deepmodel(args)
    else:
        eval_deepmodel(args)


if __name__ == "__main__":
    args = ConfigArgs()
    run(args)
    os.system('notify-send "Script ended" "And correctly!"')
