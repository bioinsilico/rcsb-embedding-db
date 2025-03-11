import time

import numpy as np

from pymilvus import (
    FieldSchema, CollectionSchema, DataType, MilvusClient
)


class EmbeddingLoader:

    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    BATCH_SIZE = 2000

    def __init__(
            self,
            collection_name,
            dim
    ):
        self.collection_name = collection_name
        self.dim = dim
        self.client = None
        self.__connect()
        self.__set_collection()

    def __connect(
            self,
            host='localhost',
            port='19530'
    ):
        self.client = MilvusClient(
            uri=f"http://{host}:{port}",
            db_name="default"
        )

    def __set_collection(self):
        id_field = FieldSchema(
            name=self.ID_FIELD,
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=100  # Adjust max_length based on your identifier length
        )

        embedding_field = FieldSchema(
            name=self.EMBEDDING_FIELD,
            dtype=DataType.FLOAT16_VECTOR,
            dim=self.dim
        )

        collection_schema = CollectionSchema(
            fields=[id_field, embedding_field],
            description="Collection storing embeddings with cosine distance."
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            collection_schema=collection_schema
        )

    def create_embedding_collection(self):
        if self.collection_name in self.client.list_collections():
            self.client.drop_collection(self.collection_name)
        self.__set_collection()

    def insert_df(self, df):
        if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
            raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")

        batch_size = self.BATCH_SIZE
        total_rows = len(df)
        num_batches = (total_rows + batch_size - 1) // batch_size  # Calculate the number of batches needed
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]

            ids = batch_df[self.ID_FIELD].tolist()
            embeddings = [(embedding/np.linalg.norm(embedding)).astype(np.float16) for embedding in batch_df[self.EMBEDDING_FIELD]]

            data = [{
                self.ID_FIELD: _id,
                self.EMBEDDING_FIELD: embedding
            } for _id, embedding in zip(ids, embeddings)]
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )

    def flush(self):
        self.client.flush(
            collection_name=self.collection_name
        )

    def compact_collection(self):
        print(f"Compacting collection")
        compaction_id = self.client.compact(
            collection_name=self.collection_name
        )
        while self.client.get_compaction_state(compaction_id) != "Completed":
            print(f"Waiting for compaction to complete... f{self.client.get_compaction_state(compaction_id)}")
            time.sleep(300)
        print(f"Collection compacted")

    def index_collection(self, index_params=None):
        print(f"Indexing collection")
        # Create an index on the embedding field with cosine distance
        if not index_params:
            index_params = {
                "metric_type": "IP",
                "index_type": "DISKANN",  # You can choose other index types as needed
                "params": {}
            }
        self.client.create_index(
            collection_name=self.collection_name,
            field_name=self.EMBEDDING_FIELD,
            index_params=index_params
        )
        print("Index created with cosine distance metric.")

    def load_collection(self):
        print("Loadig collection")
        self.client.load_collection(
            collection_name=self.collection_name,
            replica_number=1
        )
        print("Collection loaded to memory.")
