
import numpy as np

class EmbeddingLoader:

    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    BATCH_SIZE = 2000

    def __init__(
            self
    ):
        self.collection = None

    def read_df(self, df):
        if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
            raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")

        batch_size = self.BATCH_SIZE
        total_rows = len(df)
        for row in df.itertuples():
            print(row)
            exit(0)


