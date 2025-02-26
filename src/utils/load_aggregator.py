import torch


def load_aggregator(checkpoint):
    if torch.cuda.is_available():
        model = torch.load(checkpoint, weights_only=False)
    else:
        model = torch.load(checkpoint, weights_only=False, map_location='cpu')
    return model.model
