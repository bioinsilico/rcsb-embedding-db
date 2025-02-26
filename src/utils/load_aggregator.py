import torch

from utils.model import ResidueEmbeddingAggregator


def load_aggregator(checkpoint):

    if torch.cuda.is_available():
        weights = torch.load(checkpoint, weights_only=True)
    else:
        weights = torch.load(checkpoint, weights_only=True, map_location='cpu')

    nn_model = ResidueEmbeddingAggregator()
    nn_model.load_state_dict(weights)

    return nn_model
