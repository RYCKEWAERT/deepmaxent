# Description: Main file to run DeepMaxent model on the Elith data
import os
import warnings

import torch

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
        # Data parameters
        self.species_nbr = 512
        self.year_to_remove = 0
        # General parameters
        self.outputdir = "results_dm_resnet_dm"
        self.dirdata = "./data/"
        self.global_seed = 42 # Instead of 42  # the answer to life the universe and everything
        set_seed(self.global_seed)
        self.list_of_seed = get_random_seed_list(1000)
        self.repeat_seed = 1
        # self.architecture ="resnet"
        self.architecture ="resnet"

        # Cross-validation step
        self.cv = False # True for cross-validation, False for final evaluation
        self.lr_tested = [0.00002,0.0002,0.002] # Learning rate tested
        self.batch_tested = [10,250] # Batch size tested
        self.hidden_nbr_tested = [1,2] # Number of hidden layers tested
        self.w_tested=[3e-4,3e-3] # Weight decay tested
        self.num_cv_blocks = (5, 5) # Number of blocks for cross-validation
        self.validation_size = 0.1 # Size of the validation set
        self.cross_validation = "blocked" # "blocked" or "random" or "stratified"
        self.validation_criteria = "AUC" # "AUC" or "loss"
        self.epoch_cv = 50 # Number of epochs for cross-validation
        
        # Final hyperparameter values
        self.learning_rate = 0.00005 #0.00001 for the best for now
        self.epoch = 20
        self.hidden_size = 250
        self.batch_size = 100
        self.TGB = True # True for using the target group background correction
        self.weight_decay = 3e-4#1e-3 # 5e-4 if mean in deepmaxentloss
        self.hidden_nbr = 2
        self.loss_options = ["deepmaxent"] # "poisson", "deepmaxent", "bce", "ce"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.mode_wandb = 'offline' # 'online' or 'offline'
        # os.environ['WANDB_SILENT'] = "true"
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
       

        
        self.trainPOfile = "data/PresenceOnlyOccurrences/GLC24-PO-metadata-train.csv"
        self.trainPAfile = "data/PresenceAbsenceSurveys/GLC24-PA-metadata-train.csv"

        pa_base = "data/landsat/GLC24-PA-train-landsat-time-series-"
        self.PA_blue_file = pa_base + "blue.csv"
        self.PA_green_file = pa_base + "green.csv"
        self.PA_red_file = pa_base + "red.csv"
        self.PA_nir_file = pa_base + "nir.csv"
        self.PA_swir1_file = pa_base + "swir1.csv"
        self.PA_swir2_file = pa_base + "swir2.csv"

        # Same for PO
        po_base = "data/landsat/GLC24-PO-train-landsat-time-series-"
        self.PO_blue_file = po_base + "blue.csv"
        self.PO_green_file = po_base + "green.csv"
        self.PO_red_file = po_base + "red.csv"
        self.PO_nir_file = po_base + "nir.csv"
        self.PO_swir1_file = po_base + "swir1.csv"
        self.PO_swir2_file = po_base + "swir2.csv"

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
