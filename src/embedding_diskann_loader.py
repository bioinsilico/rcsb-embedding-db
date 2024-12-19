import pandas as pd
import numpy as np


def write_fvecs(filename, np_array):
    """Writes a numpy array to fvecs format."""

    with open(filename, 'wb') as f:
        for vector in np_array:
            # Write the dimension of the vector
            f.write(np.array(vector.shape[0], dtype='int32').tobytes())
            # Write the vector itself
            f.write(vector.astype(np.uint8).tobytes())


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
            f.write(np.array(n_rows, dtype='int32').tobytes())
            f.write(np.array(dim, dtype='int32').tobytes())

    def add_to_bin(self, df):
        if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
            raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")
        min_val, max_val = self.min_max(df)
        with open(self.diskann_bin_file, 'ab') as f:
            for index, row in df.iterrows():
                f.write(np.round(
                    (np.array(row[self.EMBEDDING_FIELD].tolist())-min_val)*255 / (max_val-min_val)
                ).astype(np.uint8).tobytes())

    def min_max(self, df):
        df_max = 0
        df_min = 1000000
        for start in range(0, len(df), self.BATCH_SIZE):
            chunk = df.iloc[start:start + self.BATCH_SIZE]
            chunk = pd.DataFrame(chunk[self.EMBEDDING_FIELD].tolist())
            _max = chunk.max().max()
            _min = chunk.min().min()
            if _max > df_max:
                df_max = _max
            if _min < df_min:
                df_min = _min
        return df_min, df_max


