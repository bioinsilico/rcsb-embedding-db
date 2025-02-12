
from lightning_module.inference.lightning_pst_embedding_pooling import LitStructurePstEmbeddingPooling
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

    return LitStructurePstEmbeddingPooling.load_from_checkpoint(
        checkpoint,
        nn_model=nn_model
    ).model
