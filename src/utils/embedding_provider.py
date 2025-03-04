import os
from enum import Enum
import random

import numpy as np
from pymilvus import (
    connections, Collection
)

from utils.upload_structure import get_embedding_method


class EmbeddingProvider:

    HOST = 'localhost'
    PORT = '19530'
    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    LENGTH_FIELD = 'length'
    CSM_FLAG = 'is_csm'
    EMBEDDING_DIM = 1536
    N_RESULTS = 1000

    def __init__(
            self
    ):
        self.collection = {}
        self.connect()
        self.collections()
        self.embedding_model = None
        self.embedding_path = None

    def connect(self):
        connections.connect(
            host=self.HOST,
            port=self.PORT
        )

    def collections(self):
        self.collection[MilvusCollection.instance_collection] = Collection(
            name=MilvusCollection.instance_collection
        )
        self.collection[MilvusCollection.assembly_collection] = Collection(
            name=MilvusCollection.assembly_collection
        )

    def load_model(self, model_path):
        self.embedding_model = get_embedding_method(model_path)

    def set_embedding_path(self, embedding_path):
        self.embedding_path = embedding_path

    def get_by_embedding(
            self,
            collection,
            query_embedding,
            query_length,
            is_csm=True,
            n_results=100,
            param=None,
            global_similarity=False
    ):
        if param is None:
            param = {
                "metric_type": "COSINE",
                "params": {}
            }
        search_result = self.collection[collection].search(
            data=[query_embedding],
            expr=f'{self.CSM_FLAG} == False' if not is_csm else None,
            output_fields=[self.LENGTH_FIELD],
            anns_field=self.EMBEDDING_FIELD,
            limit=n_results if n_results > self.N_RESULTS else self.N_RESULTS,
            param=param
        )[0]

        if global_similarity:
            for r in search_result:
                r.distance = _global_similarity_scale(query_length, r.length, r.distance)
            search_result = sorted(
                search_result,
                key=lambda r: r.distance,
                reverse=True
            )

        return search_result[0:n_results]

    def get_by_multi_embedding(
            self,
            collection,
            query_embedding,
            is_csm=True,
            n_results=100,
            param=None
    ):
        if param is None:
            param = {
                "metric_type": "COSINE",
                "params": {}
            }
        return self.collection[collection].search(
            data=query_embedding,
            expr=f'{self.CSM_FLAG} == False' if not is_csm else None,
            anns_field=self.EMBEDDING_FIELD,
            limit=n_results,
            param=param
        )

    def get_by_id(
            self,
            collection,
            query_id
    ):
        result = self.collection[collection].query(
            expr=f'{self.ID_FIELD} == "{query_id}"',
            output_fields=[self.EMBEDDING_FIELD, self.LENGTH_FIELD],
        )
        if len(result) == 0:
            return None
        return result[0][self.EMBEDDING_FIELD], result[0][self.LENGTH_FIELD]

    def get_random_id(self):
        if self.embedding_path:
            return ".".join(random.choice(os.listdir(self.embedding_path)).split(".")[0:2])
        return self.get_by_embedding(
            collection=MilvusCollection.instance_collection,
            query_embedding=np.random.rand(self.EMBEDDING_DIM),
            is_csm=False,
            n_results=1
        )[0][0].id

    def compute_embeddings(self, structure):
        return self.embedding_model(structure)


class MilvusCollection(str, Enum):
    instance_collection = "instance_embeddings"
    assembly_collection = "assembly_embeddings"


def _global_similarity_scale(query_length, target_length, score):
    scale_factor = min(query_length, target_length) / max(query_length, target_length)
    return (scale_factor * score ** 2) ** 3
