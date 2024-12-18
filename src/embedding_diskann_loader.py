import pandas as pd

class EmbeddingLoader:

    ID_FIELD = 'id'
    EMBEDDING_FIELD = 'embedding'
    BATCH_SIZE = 2000

    def __init__(
            self,
            diskann_tmp_folder
    ):
        self.collection = None
        self.diskann_tmp_folder = diskann_tmp_folder

    def save_to_tsv(self, df, prefix):
        if not {self.ID_FIELD, self.EMBEDDING_FIELD}.issubset(df.columns):
            raise ValueError(f"DataFrame must contain '{self.ID_FIELD}' and '{self.EMBEDDING_FIELD}' columns.")
        with open(f"{self.diskann_tmp_folder}/{prefix}.tsv", "w") as f:
            for start in range(0, len(df), self.BATCH_SIZE):
                chunk = df.iloc[start:start + self.BATCH_SIZE]
                pd.DataFrame(chunk[self.EMBEDDING_FIELD].tolis()).to_csv(f, sep="\t", index=False, header=False)

