import torch

from utils.model import TransformerEmbeddingCosine


def load_aggregator(checkpoint):

    if torch.cuda.is_available():
        weights = torch.load(checkpoint, weights_only=True)
    else:
        weights = torch.load(checkpoint, weights_only=True, map_location='cpu')

    nn_model = TransformerEmbeddingCosine(
        input_features=1536,
        nhead=12,
        num_layers=6,
        dim_feedforward=3072,
        hidden_layer=1536,
        res_block_layers=12
    )

    nn_model.load_state_dict(weights)
    return nn_model
