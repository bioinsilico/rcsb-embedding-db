
from lightning_module.inference.embedding_inference import LitEmbeddingInference
from networks.transformer_nn import TransformerEmbeddingCosine


def load_aggregator(checkpoint):
    nn_model = TransformerEmbeddingCosine(
        input_features=1536,
        nhead=12,
        num_layers=6,
        dim_feedforward=3072,
        hidden_layer=1536,
        res_block_layers=12
    )

    return LitEmbeddingInference.load_from_checkpoint(
        checkpoint,
        nn_model=nn_model
    ).model
