import torch


def load_aggregator(checkpoint):
    model = torch.load(checkpoint, weights_only=True)
    return model
