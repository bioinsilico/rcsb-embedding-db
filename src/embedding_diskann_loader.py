import numpy as np


class EmbeddingLoader:

    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    BATCH_SIZE = 20000

    def __init__(
            self,
            diskann_tmp_folder,
            total_len,
            dim,
            df_max,
            df_min
    ):
        self.collection = None
        self.diskann_bin_file = f'{diskann_tmp_folder}/embeddings.bin'
        self.df_max = df_max
        self.df_min = df_min
        self.open_bin(total_len, dim)

    def open_bin(self, n_rows, dim):
        with open(self.diskann_bin_file, 'wb') as f:
            f.write(n_rows.to_bytes(4))
            f.write(dim.to_bytes(4))

    def add_to_bin(self, df):
        if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
            raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")
        min_val, max_val = self.df_min, self.df_max
        with open(self.diskann_bin_file, 'ab') as f:
            for index, row in df.iterrows():
                f.write(np.round(
                    (np.array(row[self.EMBEDDING_FIELD].tolist())-min_val)*511 / (max_val-min_val)
                ).astype(np.uint8).tobytes())
