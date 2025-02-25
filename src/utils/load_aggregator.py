import torch


def load_aggregator(checkpoint):
    if torch.cuda.is_available():
        model = torch.load(checkpoint, weights_only=True)
    else:
        model = torch.load(checkpoint, weights_only=True, map_location='cpu')
    return model
