import numpy as np
import struct


class EmbeddingLoader:

    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    BATCH_SIZE = 20000

    def __init__(
            self,
            diskann_tmp_folder,
            total_len,
            dim
    ):
        self.collection = None
        self.diskann_bin_file = f'{diskann_tmp_folder}/embeddings.bin'
        self.open_bin(total_len, dim)

    def open_bin(self, n_rows, dim):
        with open(self.diskann_bin_file, 'wb') as f:
            f.write(n_rows.to_bytes(4))
            f.write(struct.pack('<i', n_rows))
            f.write(struct.pack('<i', dim))

    def add_to_bin(self, df):
        if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
            raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")
        with open(self.diskann_bin_file, 'ab') as f:
            for index, row in df.iterrows():
                f.write(np.array(row[self.EMBEDDING_FIELD].tolist(), dtype=np.float32).tobytes())
