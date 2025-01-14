# Description: Main file to run DeepMaxent model on the Elith data
#
import warnings
from librairies.train_models_elith import cv_eval_deepmodel, eval_deepmodel
from librairies.utils import set_seed, get_random_seed_list, make_results_directory
import os

class ConfigArgs:
    def __init__(self, config_file=None):
        if config_file:
            self.load_configuration(config_file)
        else:
            self.load_default_configuration()

    def load_default_configuration(self):
        # General parameters
        self.outputdir = "results"
        self.dirdata = "./data/elith/"
        self.global_seed = 42  # the answer to life the universe and everything
        set_seed(self.global_seed)
        self.list_of_seed = get_random_seed_list(1000)
        self.repeat_seed = 1

        # Cross-validation step
        self.cv = True # True for cross-validation, False for final evaluation
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
        self.learning_rate = 0.0002
        self.epoch = 100
        self.hidden_size = 250
        self.batch_size = 250
        self.TGB = True # True for using the target group background correction
        self.weight_decay = 3e-4 # 5e-4 if mean in deepmaxentloss
        self.hidden_nbr = 2
        self.loss_options = ["deepmaxent"] # "poisson", "deepmaxent", "bce", "ce"


        self.mode_wandb = 'offline' # 'online' or 'offline'
        os.environ['WANDB_SILENT'] = "true"
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
       
        self.regions = ["AWT", "CAN", "NSW", "NZ", "SA", "SWI"]
        # self.regions = ["CAN"]
        
        self.categoricalvars = ["ontveg", "vegsys"]#  "toxicats", "age", "calc" are ordinals/or binary
        
def disable_warnings():
    warnings.filterwarnings("ignore")
 
def run(args):

    disable_warnings()  # Ignore all warnings during the code's execution
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
