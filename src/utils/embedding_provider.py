import os
import struct
from enum import Enum
import random

import numpy as np
from pymilvus import MilvusClient


from utils.upload_structure import get_embedding_method


class EmbeddingProvider:

    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    LENGTH_FIELD = 'length'
    CSM_FLAG = 'is_csm'
    EMBEDDING_DIM = 1536
    N_RESULTS = 1000

    def __init__(
            self
    ):
        self.af_client = None
        self.rcsb_client = None
        self.collection = {}
        self.embedding_model = None
        self.embedding_path = None

    def connect(
            self,
            rcsb_host,
            afdb_host
    ):
        self.rcsb_client = MilvusClient(
            uri=f"http://{rcsb_host}:19530",
            db_name="default"
        )
        self.af_client = MilvusClient(
            uri=f"http://{afdb_host}:19530",
            db_name="default"
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
            global_similarity=False,
            output_fields=None
    ):
        if param is None:
            param = {
                "metric_type": "COSINE",
                "params": {}
            }
        limit = n_results if n_results > self.N_RESULTS else self.N_RESULTS
        expr = f'{self.CSM_FLAG} == False' if not is_csm else None
        client = self.rcsb_client
        if collection == MilvusCollection.af_collection:
            expr = None
            output_fields = None
            limit = n_results
            param = {
                "search_list": limit
            }
            client = self.af_client

        search_result = client.search(
            collection_name=collection,
            data=[query_embedding],
            filter=expr,
            output_fields=output_fields,
            anns_field=self.EMBEDDING_FIELD,
            limit=limit,
            search_params=param
        )[0]

        if global_similarity:
            for r in search_result:
                r['distance'] = _global_similarity_scale(query_length, r['entity']['length'], r['distance'])
            search_result = sorted(
                search_result,
                key=lambda r: r['distance'],
                reverse=True
            )

        return search_result[0:n_results]

    def get_by_multi_embedding(
            self,
            collection,
            query_embedding_list,
            n_results=100,
            param=None
    ):
        client = self.rcsb_client
        if param is None:
            param = {
                "metric_type": "COSINE",
                "params": {}
            }
        if collection == MilvusCollection.af_collection:
            limit = n_results
            param = {
                "search_list": limit
            }
            client = self.af_client

        return client.search(
            collection_name=collection,
            data=query_embedding_list,
            filter=None,
            output_fields=None,
            anns_field=self.EMBEDDING_FIELD,
            limit=n_results,
            search_params=param
        )

    def get_by_id(
            self,
            collection,
            query_id
    ):
        client = self.af_client if collection == MilvusCollection.af_collection else self.rcsb_client
        output_fields = [self.EMBEDDING_FIELD] if collection == MilvusCollection.af_collection else [self.EMBEDDING_FIELD, self.LENGTH_FIELD]
        result = client.query(
            collection_name=collection,
            filter=f'{self.ID_FIELD} == "{query_id}"',
            output_fields=output_fields
        )
        if len(result) == 0:
            return None, 0
        if collection == MilvusCollection.af_collection:
            return binary_to_float_np_array(result[0][self.EMBEDDING_FIELD][0]), 0.
        return np.array(result[0][self.EMBEDDING_FIELD]), result[0][self.LENGTH_FIELD]

    def get_random_id(self):
        if self.embedding_path:
            return ".".join(random.choice(os.listdir(self.embedding_path)).split(".")[0:2])
        return self.get_by_embedding(
            collection=MilvusCollection.instance_collection,
            query_embedding=np.random.rand(self.EMBEDDING_DIM),
            query_length=1,
            is_csm=False,
            n_results=1,
            global_similarity=False
        )[0]['id']

    def compute_embeddings(self, structure):
        return self.embedding_model(structure)


class MilvusCollection(str, Enum):
    instance_collection = "instance_embeddings"
    assembly_collection = "assembly_embeddings"
    af_collection = "af_embeddings"


def _global_similarity_scale(query_length, target_length, score):
    scale_factor = min(query_length, target_length) / max(query_length, target_length)
    return (scale_factor * score ** 3) ** (1/4)


def binary_to_float_np_array(binary_data):
    """
    Converts binary data to a list of float16 values.

    Args:
        binary_data: A bytes-like object containing binary data.

    Returns:
        A list of float16 values.
    """
    float16_list = []
    for i in range(0, len(binary_data), 2):  # Step size of 2 for float16 (2 bytes)
        try:
            # Unpack the binary data as a short (2 bytes) and interpret as float16
            float16_value = np.frombuffer(struct.pack('H', struct.unpack('<H', binary_data[i:i+2])[0]), dtype=np.float16)[0]
            float16_list.append(float16_value)
        except struct.error:
            print(f"Warning: Not enough data to unpack at index {i}. Skipping.")
            break
    return np.array(float16_list)
