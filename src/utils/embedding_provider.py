from enum import Enum

import numpy as np
from pymilvus import (
    connections, Collection
)


class EmbeddingProvider:

    HOST = '132.249.213.96'
    PORT = '19530'
    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    CSM_FLAG = 'is_csm'
    EMBEDDING_DIM = 1536

    def __init__(
            self
    ):
        self.collection = {}
        self.connect()
        self.collections()


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

    def get_by_embedding(
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
            data=[query_embedding],
            expr=f'{self.CSM_FLAG} == False' if not is_csm else None,
            anns_field=self.EMBEDDING_FIELD,
            limit=n_results,
            param=param
        )

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
            output_fields=[self.EMBEDDING_FIELD]
        )
        if len(result) == 0:
            return None
        return result[0][self.EMBEDDING_FIELD]

    def get_random(self):
        return self.get_by_embedding(
            collection=MilvusCollection.instance_collection,
            query_embedding=np.random.rand(self.EMBEDDING_DIM),
            is_csm=False,
            n_results=1
        )


class MilvusCollection(str, Enum):
    instance_collection = "instance_embeddings"
    assembly_collection = "assembly_embeddings"
