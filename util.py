import torch


def get_checkpoint_path():
    """Return the path to save the best performing model checkpoint.

    Parameters:
        algo (str)
          Indicates which algorithm will be used to train the model

    Returns:
        checkpoint_path (str)
            The path to save the best performing model checkpoint
    """
    return 'best_model_checkpoint.pth'

def load_model_checkpoint(checkpoint_path, model):
    """Load a model checkpoint from disk.

    Parameters:
        checkpoint_path (str)
            The path to load the checkpoint from

    Returns:
        model (torch.nn.Module)
            The model loaded from the checkpoint
    """
    model.load_state_dict(torch.load(checkpoint_path))
    return model