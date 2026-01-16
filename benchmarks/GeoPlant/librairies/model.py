import torch
import torch.nn as nn
import torchvision.models as models



class ModifiedResNet18(nn.Module):
    def __init__(self, batch_size, num_classes):
        super(ModifiedResNet18, self).__init__()

        self.norm_input = nn.LayerNorm([6,4,21])
        self.resnet18 = models.resnet18(weights=None)
        # We have to modify the first convolutional layer to accept 4 channels instead of 3
        self.resnet18.conv1 = nn.Conv2d(
            6, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet18.maxpool = nn.Identity()
        self.ln = nn.LayerNorm(1000)
        self.fc1 = nn.Linear(1000, 2056)
        self.fc2 = nn.Linear(2056, num_classes)

    def forward(self, x):
        x = self.norm_input(x)
        x = self.resnet18(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def save_mlp_model(args, model):
    """
    Save the MLP model to a PyTorch model file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): MLP model.
    """
    mlp_filepath = (
        args.outputdir + "model/MLP.pth"
    )  # Define the file path to save the MLP model file
    torch.save(model, mlp_filepath)  # Save the MLP model to a PyTorch model file


def load_mlp_model(args, model):
    """
    Load the pre-trained MLP model.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): The MLP model object.

    Returns:
        models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel: The loaded pre-trained MLP model.
    """
    mlp_filepath = (
        args.outputdir + "model/MLP.pth"
    )  # Filepath of the pre-trained MLP model
    model.load_state_dict(
        torch.load(mlp_filepath)
    )  # Load the weights of the pre-trained model
    return model


def make_predictions(model, X_tensor):
    """
    Make predictions using the given PyTorch model and input tensor.

    Parameters:
        model (torch.nn.Module): The PyTorch model.
        X_tensor (torch.Tensor): The input tensor for making predictions.

    Returns:
        torch.Tensor: The predictions.
    """
    model.eval()
    model = model.to("cpu")
    X_tensor = X_tensor.to("cpu")

    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions